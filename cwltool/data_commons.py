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
        datestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")
        jobname = "datacommonscwl-" + datestring
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

    """
    def collect_output(self, schema, builder, outdir, fs_access, compute_checksums=True):
        r = []
        if "outputBinding" in schema:
            binding = schema["outputBinding"]
            globpatterns = []

            from .utils import aslist
            if "glob" in binding:
                with SourceLine(binding, "glob", WorkflowException):
                    for gb in aslist(binding["glob"]):
                        gb = builder.do_eval(gb)
                        if gb:
                            globpatterns.extend(aslist(gb))

                    _logger.info("globpatterns: %s" % str(globpatterns))

                    for gb in globpatterns:
                        if gb.starswith(outdir):
                            gb = gb[len(outdir) + 1:]
                        elif gb == ".":
                            gb = outdir
                        elif gb.starswith("/"):
                            raise WorkflowException("glob patterns must not start with '/'")

                        r.extend([
                            {
                                "location": g,
                                "path": fs_access.join(builder.outdir, g[len(prefix[0])+1:])
                            }
                            for g in fs_acess.glob(fs_access.join(outdir, gb))
                        ])
        return r
    """

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
            shortio = {shortname(k): v for k, v in io}

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

        #TODO set chronos job dependencies based on input/output files of workflow steps
        #kferriter
        # refactor into set_job_depdencies
        """
        i = 0
        ordered_steps = []
        stepscopy = copy.deepcopy(self.steps)
        for step in self.steps:
            print("DETERMINING DEPENDENCY LINKS")
            print("step {} input: {}".format(i, step.tool["inputs"]))
            print("step {} output: {}".format(i, step.tool["outputs"]))
            i += 1
            for otherstep in stepscopy:
                if otherstep == step:
                    continue
                for inp in self.tool["inputs"]:
                    pass
                    #print(inp)
        """
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


"""
When the cwl document is a workflow with multiple steps, check the inputs and outputs.
If one job A takes as an input an output of another step B, set step B as a
parent of job A.

Takes an iterable list of jobs, returns a copy of that list of jobs, with parent fields set.
"""
def set_job_dependencies(jobs):
    joblist = list(jobs)
    for job in joblist:
        #pprint(vars(job))
        job_order_object = job.joborder
        #TODO


    return joblist

"""
Return a string containing a table of upcoming n jobs
"""
def show_upcoming_jobs(count=10):
    stars_client = Stars(
        services_endpoints  = ["https://stars-app.renci.org/marathon"],
        scheduler_endpoints = ["stars-app.renci.org/chronos"])
    jobs = stars_client.scheduler.list()

    i = 0
    headers = ["Job Name", "Next Scheduled Run"]
    table = []
    for j in jobs:
        name = j["name"]
        (repeat, start, interval) = j["schedule"].split("/")
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
                    interval_count = elapsed / interval
                    frac = interval_count % 1
                    time_to_next = frac * interval
                    next_time = now + time_to_next
        else:
            # created with no start time, defaulting to now (will be filled in by chronos later)
            next_time = now
        table.append([name, str(next_time)])
    table.sort(key=lambda v: v[1])
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
Send the commands to the data commons API
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
    pivot = Stars(
        services_endpoints  = ["https://stars-app.renci.org/marathon"],
        scheduler_endpoints = ["stars-app.renci.org/chronos"])

    command = " ".join(commands)
    if stdin:
        command = command + " < " + stdin
    if stdout:
        command = command + " > " + stdout
    if stderr:
        command = command + " 2> " + stderr

    if container_command:
        command = container_command + " " + command

    kwargs = {
        "name":jobname,
        "command":command,
        "owner":"ted@job.org",
        "runAsUser":"evryscope",
        "schedule":"R//PT60M",
        #"constraints":[
        #    [
        #        "hostname",
        #        "EQUALS",
        #        "stars-dw5.edc.renci.org"
        #    ]
        #],
        "execute_now":True,
        "shell":True
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
