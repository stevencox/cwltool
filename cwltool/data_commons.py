import argparse
import sys
import os
import copy
import json
import datetime
import uuid
import functools
import re
import isodate
import tabulate
from pprint import pprint
from typing import Any, Callable, Dict, List, Text, Union, cast
from functools import partial
from collections import namedtuple

import cwltool.load_tool
import cwltool.resolver
import cwltool.draft2tool
import cwltool.main
import cwltool.process
import cwltool.pathmapper
import cwltool.workflow
import cwltool.expression
from cwltool.flatten import flatten
from cwltool.errors import WorkflowException
from cwltool.utils import aslist

from schema_salad.ref_resolver import uri_file_path
from schema_salad.sourceline import SourceLine, indent

from stars.stars import Stars

import logging
_logger = logging.getLogger("datacommons")
_logger.setLevel(logging.INFO)

WorkflowStateItem = namedtuple("WorkflowStateItem", ["parameter", "value", "success"])

def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow_file", type=str,
            help="CWL workflow specification file")
    parser.add_argument("job_file", nargs='+',
            help="One or more whitespace separated job files")

    # directory to write any outputs to
    # defaults to the current working directory
    parser.add_argument("--outdir", type=str, default=os.getcwd(),
            help="Directory to write the outputs to. "\
                "Defaults to the current working directory.")

    # if executed from command line, update args to those
    if args is None:
        args = sys.argv[1:]

    options = parser.parse_args(args)
    print("Options: " + str(options))

    # create a cwltool object from the cwl workflow file
    try:
        tool = cwltool.load_tool.load_tool(
            options.workflow_file,
            makeDataCommonsTool,
            kwargs={},
            resolver=cwltool.resolver.tool_resolver
        )
        print("Tool:")
        print(vars(tool))
    except cwltool.process.UnsupportedRequirement as e:
        print("UnsupportedRequirement")

    # set up args for load_job_order
    options.workflow = options.workflow_file
    options.job_order = options.job_file

    # set default basedir to be the cwd.
    #Maybe set to None and let load_job_order determine it
    options.basedir = os.getcwd()
    options.tool_help = None
    options.debug = True
    # load the job files
    job, _ = cwltool.main.load_job_order(options, tool, sys.stdin)
    print("Job: ")
    pprint(job)
    for inputname in job:
        print("inputname: {}".format(inputname))
        if inputname == "file":
            filearg = job["file"]
            print("filearg: {}".format(filearg))
            if filearg.get("location"):

                filearg["path"] = uri_file_path(filearg["location"])

    kwargs = {
        'basedir': options.basedir,
        'outdir': options.basedir
    }
    jobiter = tool.job(job, None, **kwargs)

    for jobrunner in jobiter:
        if jobrunner:
            jobrunner.run(**kwargs)
        else:
            print("")


"""
Create and return an object that wraps cwltool.job.CommandLineJob
"""
def makeJob(tool, job, **kwargs):
    pass


"""
Subclass cwltool.job.CommandLineJob and simplify behavior
"""
class DataCommonsCommandLineJob(cwltool.job.CommandLineJob):
    def __init__(self, **kwargs):
        super().__init__()
        self.container_command = None

    """
    Overriding job setup to ignore local path resolutions
    """
    def _setup(self, kwargs):
        pass

    def run(self, pull_image=True, rm_container=True,
            rm_tmpdir=True, move_outputs="move", **kwargs):
        # type: (bool, bool, bool, Text, **Any) -> None
        self._setup(kwargs)
        # Not sure preserving the environment is necessary, as the job is not executing locally
        # might want to send the environment to the Chronos API though
        env = self.environment
        self._execute([], env, rm_tmpdir=rm_tmpdir, move_outputs=move_outputs)

    """
    Create the command string and run it, and print results
    """
    def _execute(self, runtime, env, rm_tmpdir=True, move_outputs="move"):
        _logger = logging.getLogger("datacommons")

        # pruned from cwltool.job.JobBase._execute
        shouldquote = lambda x: False

        outputs = {}  # type: Dict[Text,Text]

        try:
            commands = [Text(x) for x in (runtime + self.command_line)]

            #print("Commands: " + str(commands))
            rcode = _datacommons_popen(
                self.name,
                commands,
                env=env,
                cwd=self.outdir,
                container_command=self.container_command,
                stdin=self.stdin,
                stdout=self.stdout,
                stderr=self.stderr
            )

            if self.successCodes and rcode in self.successCodes:
                processStatus = "success"
            elif self.temporaryFailCodes and rcode in self.temporaryFailCodes:
                processStatus = "temporaryFail"
            elif self.permanentFailCodes and rcode in self.permanentFailCodes:
                processStatus = "permanentFail"
            elif rcode == 0:
                processStatus = "success"
            else:
                processStatus = "permanentFail"

        except OSError as e:
            if e.errno == 2:
                if runtime:
                    _logger.warn(u"'%s' not found", runtime[0])
                else:
                    _logger.warn(u"'%s' not found", self.command_line[0])
            else:
                _logger.warn("Exception while running job")
            processStatus = "permanentFail"
        except WorkflowException as e:
            _logger.warn(u"[job %s] Job error:\n%s" % (self.name, e))
            processStatus = "permanentFail"
        except Exception as e:
            _logger.warn("Exception while running job: " + str(e))
            #import traceback
            #traceback.print_tb(e.__traceback__)
            processStatus = "permanentFail"

        if processStatus != "success":
            _logger.warn(u"[job {}] completed {}".format(self.name, processStatus))
        else:
            _logger.debug(u"[job {}] completed {}".format(self.name, processStatus))

        #print(u"[job {}] {}".format(self.name, json.dumps(outputs, indent=4)))

        """
        # evaluate expressions in the outputs field
        pprint(vars(self.tool.tool))
        r = []
        output_schema = self.tool.tool["outputs"]
        print("output_schema: " + str(output_schema))
        for output in output_schema:
            print("output: " + str(output))
            fragment = cwltool.process.shortname(output["id"])
            if "outputBinding" in output:
                binding = output["outputBinding"]
                globpatterns = []
                if "glob" in binding:
                    for gb in aslist(binding["glob"]):
                        gb = self.builder.do_eval(gb)
                        if gb:
                            globpatterns.extend(aslist(gb))

                    outdir = self.outdir
                    fs_access = self.builder.make_fs_access(outdir)
                    for gb in globpatterns:

                        prefix = fs_access.glob(builder.outdir)
                        r.extend([{"location": g,
                                   "path": fs_access.join(builder.outdir, g[len(prefix[0])+1:]),
                                   "basename": os.path.basename(g),
                                   "nameroot": os.path.splitext(os.path.basename(g))[0],
                                   "nameext": os.path.splitext(os.path.basename(g))[1],
                                   "class": "File" if fs_access.isfile(g) else "Directory"}
                                  for g in fs_access.glob(fs_access.join(outdir, gb))])
        print("r: " + str(r))
        """
        #outputs = self.collect_outputs(self.outdir)
        outputs = self.collect_outputs("/renci/irods")
        # Maybe want some sort of callback
        self.output_callback(outputs, processStatus)


"""
Subclass cwltool.job.CommandLineJob (DockerCommandLineJob)
"""
class DataCommonsDockerCommandLineJob(DataCommonsCommandLineJob):

    def run(self, pull_image=True, rm_container=True,
            rm_tmpdir=True, move_outputs="move", **kwargs):
        docker_req, docker_is_req = cwltool.process.get_feature(self, "DockerRequirement")

        # maybe pass local environment over in future
        self._setup(kwargs)
        env = self.environment

        if hasattr(self, "requirements"):
            for req in self.requirements:
                if req["class"] == "DockerRequirement":
                    #print("DockerRequirement")
                    #TODO add functionality for other docker* fields
                    if "dockerPull" in req:
                        image_tag = req["dockerPull"]
                        #print("dockerPull: {}".format(image_tag))
                    else:
                        raise WorkflowException("DockerRequirement specified without image tag")

        self.container_command = \
            "docker run --rm -v /renci/irods:/irods " + str(image_tag) + " "

        self._execute([], env, rm_tmpdir=rm_tmpdir, move_outputs=move_outputs)


"""
Make a tool object from a loaded cwl workflow/tool object
"""
def makeDataCommonsTool(cwl_obj, **kwargs):
    # not a cwl object, so stop
    if not isinstance(cwl_obj, dict):
        raise WorkflowException("CWL object not a dict {}".format(cwl_obj))
    if cwl_obj.get("class") == "CommandLineTool":
        return DataCommonsCommandLineTool(cwl_obj, **kwargs)
    elif cwl_obj.get("class") == "Workflow":
        return cwltool.workflow.Workflow(cwl_obj, **kwargs)
        #return DataCommonsWorkflow(cwl_obj, **kwargs)
    else:
        raise WorkflowException("Unsupported CWL class type : {}".format(cwl_obj.get("class")))


"""
Subclass of the cwltool CommandLineTool to override path mapping
"""
class DataCommonsCommandLineTool(cwltool.draft2tool.CommandLineTool):

    def makeJobRunner(self, **kwargs):
        dockerReq, _ = self.get_requirement("DockerRequirement")
        # don't support the forced --use-container flag, or --no-container
        if dockerReq:
            return DataCommonsDockerCommandLineJob()
        else:
            return DataCommonsCommandLineJob()

    def makePathMapper(self, reffiles, stagedir, **kwargs):
        #return super().makePathMapper(reffiles, stagedir, **kwargs)
        pass

    def job(self, job_order, output_callback, **kwargs):
        # modified from cwltool.draft2tool.CommandLineTool.job
        #jobname = uniquename(kwargs.get("name", shortname(self.tool.get("id", "job"))))
        tool_name = cwltool.process.shortname(self.tool.get("id"))
        #print("tool name: {}".format(self.tool.get("id")))
        datestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        jobname = "datacommonscwl-{}-{}".format(datestring, tool_name)
        builder = self._init_job(job_order, **kwargs)

        reffiles = copy.deepcopy(builder.files)
        j = self.makeJobRunner(**kwargs)
        j.builder = builder
        j.joborder = builder.job
        j.make_pathmapper = self.makePathMapper
        j.stdin = None
        j.stderr = None
        j.stdout = None
        j.successCodes = self.tool.get("successCodes")
        j.temporaryFailCodes = self.tool.get("temporaryFailCodes")
        j.permanentFailCodes = self.tool.get("permanentFailCodes")
        j.requirements = self.requirements
        j.hints = self.hints
        j.name = jobname
        j.output_callback = output_callback
        j.tool = self
        j.outdir = kwargs.get("outdir", ".")
        j.basedir = kwargs.get("basedir", ".")
        j.outdir = "/renci/irods"
        j.basedir = "/renci/irods"
        #kferriter
        #print("j.outdir: {}".format(j.outdir))
        #print("j.basedir: {}".format(j.basedir))

        builder.pathmapper = None
        make_path_mapper_kwargs = kwargs
        if "stagedir" in make_path_mapper_kwargs:
            make_path_mapper_kwargs = make_path_mapper_kwargs.copy()
            del make_path_mapper_kwargs["stagedir"]

        # possibly remove
        builder.pathmapper = self.makePathMapper(reffiles, builder.stagedir, **make_path_mapper_kwargs)
        builder.requirements = j.requirements

        # convert these into command line stdin/stdout/stderr stream redirection
        """
        if self.tool.get("stdin"):
            with SourceLine(self.tool, "stdin", validate.ValidationException):
                j.stdin = builder.do_eval(self.tool["stdin"])
                reffiles.append({"class": "File", "path": j.stdin})

        if self.tool.get("stderr"):
            with SourceLine(self.tool, "stderr", validate.ValidationException):
                j.stderr = builder.do_eval(self.tool["stderr"])
                if os.path.isabs(j.stderr) or ".." in j.stderr:
                    raise validate.ValidationException("stderr must be a relative path, got '%s'" % j.stderr)

        if self.tool.get("stdout"):
            with SourceLine(self.tool, "stdout", validate.ValidationException):
                j.stdout = builder.do_eval(self.tool["stdout"])
                if os.path.isabs(j.stdout) or ".." in j.stdout or not j.stdout:
                    raise validate.ValidationException("stdout must be a relative path, got '%s'" % j.stdout)
        """
        if self.tool.get("stdin"):
            j.stdin = builder.do_eval(self.tool.get("stdin"))
        if self.tool.get("stderr"):
            j.stderr = builder.do_eval(self.tool.get("stderr"))
        if self.tool.get("stdout"):
            j.stdout = builder.do_eval(self.tool.get("stdout"))


        #print(u"[job {}] command line bindings is {}".format(j.name, json.dumps(builder.bindings, indent=4)))

        def locToPath(p):
            if "path" not in p and "location" in p:
                p["path"] = uri_file_path(p["location"])
                #del p["location"]
        # change "Location" field on file class to "Path"
        cwltool.pathmapper.visit_class(builder.bindings, ("File","Directory"), locToPath)

        j.command_line = flatten(list(map(builder.generate_arg, builder.bindings)))
        j.pathmapper = builder.pathmapper
        """
        j.collect_outputs = partial(
            super().collect_output_ports, self.tool["outputs"], builder,
            compute_checksum=kwargs.get("compute_checksum", True),
            jobname=jobname,
            readers=None)
        """

        j.collect_outputs = partial(self.collect_output_ports, self.tool["outputs"], builder)
        yield j

    def collect_output_ports(self, ports, builder, outdir):
        ret = {}
        for i, port in enumerate(ports):
            #print("port: {}".format(port))
            with SourceLine(ports, i, WorkflowException):
                fragment = cwltool.process.shortname(port["id"])
                try:
                    ret[fragment] = self.collect_output(port, builder, outdir)
                except Exception as e:
                    _logger.debug("Error collecting output for '{}'".format(port["id"]))
                    raise WorkflowException("Error collecting output for '{}'".format(port["id"]))
        _logger.debug("collect_output_ports finished: {}".format(ret))
        return ret

    def collect_output(self, schema, builder, outdir):
        r = []
        if "outputBinding" in schema:
            binding = schema["outputBinding"]
            globpatterns = []

            #revmap = partial(revmap_file, builder, outdir)

            if "glob" in binding:
                for gb in aslist(binding["glob"]):
                    gb = builder.do_eval(gb)
                    _logger.debug("gb evaluated to: '{}'".format(gb))
                    if gb:
                        globpatterns.extend(aslist(gb))

                for gb in globpatterns:
                    g = os.path.join(outdir, gb)
                    cls = "File" if schema["type"] == "File" else "Directory"

                    r.extend([
                        {"location": g,
                         "path": g,
                         "basename": os.path.basename(g),
                         "nameroot": os.path.splitext(os.path.basename(g))[0],
                         "nameext": os.path.splitext(os.path.basename(g))[1],
                         "class": cls}

                    ])
        optional = False
        single = False
        if isinstance(schema["type"], list):
            if "null" in schema["type"]:
                optional = True
            if "File" in schema["type"] or "Directory" in schema["type"]:
                single = True
        elif schema["type"] == "File" or schema["type"] == "Directory":
            single = True
        if "outputEval" in binding:
            r = builder.do_eval(binding["outputEval"], context=r)

        if single:
            if not r and not optional:
                raise WorkflowException("Did not find output file with glob pattern: '{}'".format(globpatterns))
            elif not r and optional:
                pass
            elif isinstance(r,list):
                if len(r) > 1:
                    raise WorkflowException("Multiple matches for output item that is a single file")
                else:
                    r = r[0]

        #print("collect_output finished: {}".format(r))
        return r


class DataCommonsPathMapper(cwltool.pathmapper.PathMapper):
    def __init__(self, referenced_files, basedir):
        self._pathmap = {}
        self.stagedir = basedir
        super().setup(dedup(referenced_files), basedir)


class DataCommonsWorkflow(cwltool.workflow.Workflow):
    def __init__(self, toolpath_object, **kwargs):
        super(DataCommonsWorkflow, self).__init__(toolpath_object, **kwargs)

    def job(self, job_order, output_callbacks, **kwargs):
        #super(DataCommonsWorkflow, self).job(job_order, output_callbacks, **kwargs)
        builder = self._init_job(job_order, **kwargs)
        wj = DataCommonsWorkflowJob(self, **kwargs)
        yield wj
        kwargs["part_of"] = "workflow %s" % wj.name

        #TODO decide where to handle job dependency linking
        # look at inputs linked to outputs, and set jobs with dependent inputs as children
        """
        for step in self.steps:
            print("step: {}".format(step))
            print("step input: {}".format(step.tool["inputs"]))
            print("step output: {}".format(step.tool["outputs"]))
            for inp in step.tool["inputs"]:
                ev = builder.do_eval(inp["source"])
                print(ev)
        """
        for w in wj.job(builder.job, output_callbacks, **kwargs):
            yield w
    def visit(self, op):
        op(self.tool)
        for s in self.steps:
            s.visit(op)

class DataCommonsWorkflowJobStep(cwltool.workflow.WorkflowJobStep):
    def __init__(self, step):
        self.step = step
        self.tool = step.tool
        self.id = step.id
        self.name = "step " + self.id

    def job(self, joborder, output_callback, **kwargs):
        kwargs["part_of"] = self.name
        kwargs["name"] = self.name
        _logger.debug("[{}] start".format(self.name))

        for j in self.step.job(joborder, output_callback, **kwargs):
            yield j

class DataCommonsWorkflowJob(cwltool.workflow.WorkflowJob):
    def __init__(self, workflow, **kwargs):
        super().__init__(workflow, **kwargs)
        self.steps = [DataCommonsWorkflowJobStep(s) for s in workflow.steps]

    def do_output_callback(self, final_output_callback):
        #TODO
        super().do_output_callback(final_output_callback)

    def receive_output(self, step, outputparms, final_output_callback, jobout, processStatus):
        #TODO
        super().receive_output(step, outputparms, final_output_callback, jobout, processStatus)

    """
    Modifying this to stop it from checking for input file existence
    """
    def try_make_job(self, step, final_output_callback, **kwargs):
        inputparms = step.tool["inputs"]
        outputparms = step.tool["outputs"]
        #_logger.debug("[{}] inputparms: {}".format(self.name, inputparms))
        #_logger.debug("[{}] outputparms: {}".format(self.name, outputparms))

        valueFrom = {
                i["id"]: i["valueFrom"] for i in step.tool["inputs"]
                if "valueFrom" in i}

        def postScatterEval(io):
            # type: (Dict[Text, Any]) -> Dict[Text, Any]
            shortio = {cwltool.process.shortname(k): v for k, v in io}

            def valueFromFunc(k, v):  # type: (Any, Any) -> Any
                if k in valueFrom:
                    return cwltool.expression.do_eval(
                        valueFrom[k], shortio, self.workflow.requirements,
                        None, None, {}, context=v, debug=debug, js_console=js_console)
                else:
                    return v

            return {k: valueFromFunc(k, v) for k, v in io.items()}
        #TODO HANDLE SCATTER

        inputobj = object_from_state(self.state, inputparms, False, False, "source")
        #_logger.debug("inputobj: {}".format(inputobj))

        callback = functools.partial(self.receive_output, step, outputparms, final_output_callback)

        #_logger.debug("step: {}".format(self.name, step))
        jobs = step.job(inputobj, callback, **kwargs)
        for j in jobs:
            yield j

    def run(self, **kwargs):
        _logger.debug("[{}] run called".format(self.name))
        pass

    def job(self, joborder, output_callback, **kwargs):
        #print("self.tool['inputs']: {}".format(self.tool["inputs"]))
        self.state = {}
        for inp in self.tool["inputs"]:
            iid = cwltool.process.shortname(inp["id"])
            self.state[inp["id"]] = WorkflowStateItem(inp, copy.deepcopy(joborder[iid]), "success")

        for step in self.steps:
            for out in step.tool["outputs"]:
                self.state[out["id"]] = None

        for step in self.steps:
            step.iterator = self.try_make_job(step, output_callback, **kwargs)
            if step.iterator:
                for subjob in step.iterator:
                    yield subjob


def object_from_state(state, parms, frag_only, supportsMultipleInput, sourceField, incomplete=False):
    inputobj = {}
    for inp in parms:
        iid = inp["id"]
        if frag_only:
            iid = cwltool.process.shortname(iid)
        if sourceField in inp:
            connections = aslist(inp[sourceField])
            for src in connections:
                if src in state and state[src] is not None:
                    if not cwltool.workflow.match_types(
                            inp["type"], state[src], iid, inputobj,
                            inp.get("linkMerge", ("merge_nested" if len(connections)>1 else None)),
                            valueFrom=inp.get("valueFrom")):
                        raise WorkflowException("Type mismatch between source and sink")
        if inputobj.get(iid) is None and "default" in inp:
            inputobj[iid] = copy.copy(inp["default"])
        if iid not in inputobj and ("valueFrom" in inp or incomplete):
            inputobj[iid] = None
        if iid not in inputobj:
            raise WorkflowException("Value for {} not specified".format(inp["id"]))
    return inputobj


def get_stars_client():
    # store client in static function var, so there's only one instance
    if not hasattr(get_stars_client, "stars_client"):
        get_stars_client.stars_client = Stars(
            services_endpoints  = ["https://stars-app.renci.org/marathon"],
            scheduler_endpoints = ["stars-app.renci.org/chronos"])
    return get_stars_client.stars_client

"""
When the cwl document is a workflow with multiple steps, check the inputs and outputs.
If one job A takes as an input an output of another step B, set step B as a
parent of job A.

Takes an iterable list of workflow jobs.
Each job has a step attribute if it is part of a workflow
"""
def set_job_dependencies(original_jobs):
    _logger.info("Determining workflow dependency links")
    jobs = []
    # traverse through steps and simplify down the input/ouput structure
    # in/out lists for each simplified job will contain only string identifiers
    # ex:
    # jobs=[{'name':'job1', 'in':['#step1/input1'], 'out':['#step1/output1']},
    #       {'name':'job2', 'in':['#step1/output1'], 'out':['#step2/output1']}
    for j in original_jobs:
        _logger.info("compressing fields of job: {}".format(j))
        if isinstance(j, cwltool.workflow.WorkflowJob):
            # this is the wrapper job representing a workflow, skip it
            continue
        if not hasattr(j, "step") or not j.step \
                or not hasattr(j.step, "iterable") or not j.step.iterable:
            #job is not part of a workflow, cannot be dependent on other jobs
            run_job_now(j.name)
            continue
        step = j.step
        #if not hasattr(step, "iterable") or not step.iterable:
        #    # this is not a subworkflow job step, skip it. Could be the root workflow tool
        #    continue
        new_j = {}
        tool = step.tool
        j_inp = tool["in"]
        j_outp = tool["out"]

        #add needed values to new j obj
        new_j["name"] = j.name
        new_j["in"] = []
        for inp in j_inp:
            _logger.debug("step in: {}".format(inp))
            if "valueFrom" in inp and "source" not in inp:
                print("Does not currently support valueFrom. Use source")
                return
            # resource urls for the input field and the field it gets it value from
            id = inp["id"]
            source = inp["source"]
            print("id: '{}', source: '{}'".format(id, source))
            # trailing hash fragment in the resource url is the simple id
            id = id[id.rfind("#"):]


            if isinstance(source, str):
                # single source field value
                source = source[source.rfind("#"):]
                new_j["in"].append(source)
            elif isinstance(source, list):
                # using MultipleInputFeatureRequirement
                # and this field has multiple input source links
                # supports linkMerge: merge_flattened or linkMerge: merge_nested
                # TODO have tested merge_flattened, need to test merge_nestedq
                for elem in source:
                    if isinstance(elem, list):
                        for nested_elem in elem:
                            nested_elem = nested_elem[nested_elem.rfind("#"):]
                            new_j["in"].append(nested_elem)
                    else:
                        elem = elem[elem.rfind("#"):]
                        new_j["in"].append(elem)

            #new_s["id"] = id

        new_j["out"] = []
        for outp in j_outp:
            _logger.debug("step out: {}".format(outp))
            if isinstance(outp, str):
                id = outp[outp.rfind("#"):]
            elif "id" in outp:
                id = outp["id"]
                id = id[id.rfind("#"):]
            else:
                print("Out field for step is misformatted")
                return
            new_j["out"].append(id)

        jobs.append(new_j)

    # iterate through jobs. look at their inputs
    # for each input, loop through other steps and see if an output matches it
    # if it does, add the jobname of the other step to this this step's parent key
    # after each step has been checked, if the parent key has items, update in Chronos
    for j in jobs:
        j["parents"] = []
        for inp in j["in"]:
            for other_j in [other_j for other_j in jobs if other_j["name"] != j["name"]]:
                if inp in other_j["out"]:
                    j["parents"].append(other_j["name"])
    _logger.debug("dependencies determined")
    update_dependencies_in_chronos(jobs)


"""
Update a job in chronos with a schedule to run one time starting immediately
"""
def run_job_now(jobname):
    stars_client = get_stars_client()
    chronos_client = stars_client.scheduler.client
    chronos_job = chronos_client.search(name=jobname)[0]
    if "parents" in chronos_job:
        _logger.warn("Cannot update job '{}' to run now, is dependent job".format(jobname))
    chronos_job["schedule"] = "R1//P1D"
    _logger.debug("Setting job '{}' to run now".format(jobname))
    chronos_client.update(chronos_job)


"""
Take a list of dict objects. Each dict is in the format:
{'name': <jobname>, 'parents': [<parentjobname>,...]}
For each element, attempt to update a job with given name, by adding parent jobs to it.
"""
def update_dependencies_in_chronos(job_list):
    stars_client = get_stars_client()
    chronos_client = stars_client.scheduler.client
    for j in job_list:
        # get first job in chronos with this name
        chronos_job = chronos_client.search(name=j["name"])[0]
        if len(j["parents"]) == 0:
            # is a root level job, is dependent on nothing
            # set to run immediately
            chronos_job["schedule"] = "R1//P1D" #3rd segment doesn't matter with R1
        else:
            # is a child job, set parents, and set no schedule
            if "parents" not in chronos_job:
                chronos_job["parents"] = []
            for p in j["parents"]:
                if p not in chronos_job["parents"]:
                    _logger.debug("Adding {} as parent of {}".format(p, j["name"]))
                    chronos_job["parents"].append(p)
            # job has parents, so cannot have schedule field
            del chronos_job["schedule"]

        _logger.debug("Updating {} in chronos with:\n{}".format(j["name"], chronos_job))
        chronos_client.update(chronos_job)


"""
Get the time of the next chronos run in UTC, given a chronos formatted schedule string
"""
def get_next_chronos_run(schedule_str):
    if not schedule_str:
        return None

    (repeat, start, interval) = schedule_str.split("/")
    now = datetime.datetime.now(datetime.timezone.utc)
    if start:
        start = isodate.parse_datetime(start)
        if interval:
            interval = isodate.parse_duration(interval)
            elapsed_since_start = now - start
            if elapsed_since_start.total_seconds() < 0:
                # hasn't started yet
                next_time = start
            else:
                # already started, calculate next
                # number of job intervals that have passed since the job was started
                interval_count = elapsed_since_start / interval
                frac = interval_count % 1
                time_to_next = frac * interval
                next_time = now + time_to_next
    else:
        # created with no start time, defaulting to now (will be filled in by chronos later)
        next_time = now

    # localize time
    next_time = next_time.replace(tzinfo=datetime.timezone.utc).astimezone(tz=None)
    return next_time


"""
Look at job dependencies and include that in a table printout of the next [count]
jobs in chronos
#TODO implement count limiting. currently prints all jobs in chronos
#TODO it doesn't appear to be automatically sorting by next scheduled run or by creation time
"""
def show_upcoming_jobs_tree(count=20):
    stars_client = get_stars_client()
    jobs = stars_client.scheduler.list()
    headers = ["Job Name", "Next Scheduled Run", "Parent Jobs"]
    table = []
    """
    chronos only gives us child->parent relationships
    for human readability, it's better to use parent->child visual structure
    make bi-directional graph structure of parent<->child relationships
    {
        'job1': {'next_time': '2017-12-06T21:01:02.773Z', 'children': ['job2', 'job3']},
        'job2': {'next_time': '', 'children': ['job4'], 'parents': ['job1']},
        'job4': {'next_time': '', 'children': ['job5', 'job6'], 'parents': ['job1']}
    }
    """
    d = {}
    for j in jobs:
        name = j["name"]
        if "parents" not in j:
            # root level job
            if name not in d:
                d[name] = {}
            next_time = get_next_chronos_run(j["schedule"])
            d[name]["next_time"] = isodate.datetime_isoformat(next_time) if next_time else ""
        else:
            # child job.
            # add top level element to dict with parents listed
            d[name] = {"parents": j["parents"], "next_time": ""}
            #add the name of this job to the children field of
            # the parent job in the root level of the d object
            for parent_name in j["parents"]:
                if parent_name not in d:
                    d[parent_name] = {}
                if "children" not in d[parent_name]:
                    d[parent_name]["children"] = []
                d[parent_name]["children"].append(j["name"])
    #pprint(d)

    # process into a nice looking table
    # print each job, along with parents
    # if no parents, show the time it will run
    for jobname, val in d.items():
        job_row = [jobname, "", ""]
        sub_rows = []
        if "parents" in val:
            job_row[1] = "[after parents]"
            i = 0
            for parent in val["parents"]:
                if i == 0:
                    job_row[2] = parent
                else:
                    sub_rows.append(["", "", parent])
        else:
            # is root level, has a next scheduled run time
            job_row[1] = val["next_time"]
        table.append(job_row)
        table += sub_rows

        next_time = val["next_time"] or "[after parents]"

    return tabulate.tabulate(table, headers, tablefmt="grid")


"""
Return a string containing a table of the next [count] upcoming jobs
"""
def show_upcoming_jobs(count=10):
    stars_client = get_stars_client()
    jobs = stars_client.scheduler.list()

    i = 0
    headers = ["Job Name", "Next Scheduled Run"]
    table = []
    for j in jobs:
        name = j["name"]
        if "schedule" in j:
            next_time = get_next_chronos_run(j["schedule"])
        else:
            # job is a dependent
            next_time = ""
        table.append([name, str(next_time)])
    #TODO update table printout to contain info about dependent jobs
    # tree structure?
    #table.sort(key=lambda v: v[1])
    if len(table) > count:
        table = table[:count]
    return tabulate.tabulate(table, headers, tablefmt="grid")


def verify_endpoint_job(stars_client, jobname):
    # TODO update stars package to allow chronos /search functionality
    # so it is not returning all jobs into local memory
    jobs = stars_client.scheduler.list()
    for j in jobs:
        if j["name"] == jobname:
            # job was created
            (repeat, start, interval) = j["schedule"].split("/")
            repeat_match = re.compile("R(\d)").match(repeat)
            if repeat_match:
                repeat = repeat_match.group(1)
            else:
                repeat = "infinite"

            if start:
                start = isodate.parse_datetime(start)
            else:
                start = "[immediately]"

            if interval:
                interval = isodate.parse_duration(interval)

            _logger.info("Job '{}' created at endpoint(s) {}. Repeating {} times, every {}, starting at {}"
                .format(jobname, stars_client.scheduler.client.servers, repeat, interval, start))


"""
Send the commands to the data commons API.
Creates the jobs to initially run 50yrs in future, and every year after.
Start time must be reduced later via api in order for them to run before that.
"""
def _datacommons_popen(
        jobname,
        commands,  # type: List[Text]
        env,  # type: Union[MutableMapping[Text, Text], MutableMapping[str, str]]
        cwd,  # type: Text
        container_command=None, # type string
        stdin=None,
        stdout=None,
        stderr=None,
    ):

    _logger.debug("STARS: cmd: {}".format(commands))
    pivot = get_stars_client()

    command = " ".join(commands)
    if stdin:
        command = command + " < " + stdin
    if stdout:
        command = command + " > " + stdout
    if stderr:
        command = command + " 2> " + stderr

    if container_command:
        command = container_command + " " + command

    far_in_future_iso8601 = isodate.datetime_isoformat(
        datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=365*50))
    schedule = "R/{}/P1Y".format(far_in_future_iso8601)
    #2017-12-06T21:01:02.773Z

    kwargs = {
        "name": jobname,
        "command": command,
        "owner": "ted@job.org",
        "runAsUser": "evryscope",
        "schedule": schedule,
        #"constraints":[
        #    [
        #        "hostname",
        #        "EQUALS",
        #        "stars-dw5.edc.renci.org"
        #    ]
        #],
        "execute_now": False,
        "shell": True
    }

    #if container_args:
        # add container arguments to the kwargs
    #    kwargs["container"] = container_args

    pivot.scheduler.add_job(
        **kwargs
    )

    verify_endpoint_job(pivot, jobname)

    rcode = 0
    return rcode