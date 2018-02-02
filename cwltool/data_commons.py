import argparse
import sys
import os
import copy
import json
import uuid
import functools
import re
import isodate
import dateutil.parser
import datetime
import pytz
import time
import tabulate
import six
import chronos
from pprint import pprint
from typing import Any, Callable, Dict, List, Text, Union, cast
from functools import partial
from collections import namedtuple

import cwltool.load_tool
import cwltool.resolver
import cwltool.draft2tool
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
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

WorkflowStateItem = namedtuple("WorkflowStateItem", ["parameter", "value", "success"])

# store job json descriptions locally
job_cache = []

def isoformat(datetime):
    return datetime.isoformat().replace(" ", "T")

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

    # delayed import cwltool.main (circular dependency)
    import cwltool.main

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
Subclass cwltool.job.CommandLineJob and simplify behavior
"""
class DataCommonsCommandLineJob(cwltool.job.CommandLineJob):
    def __init__(self, **kwargs):
        super(DataCommonsCommandLineJob, self).__init__()
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
                jobname=self.name,
                commands=commands,
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


        # we do not use a separate outdir right now.  All files are assumed to be
        # in the same basedir in the shared filesystem mount, including output files
        outputs = self.collect_outputs(self.basedir)

        # TODO allow for outdir specification. require it to be in the shared filesystem
        # https://github.com/stevencox/cwltool/issues/10
        #outputs = self.collect_outputs(self.outdir)

        self.output_callback(outputs, processStatus)


"""
Subclass cwltool.job.CommandLineJob (DockerCommandLineJob)
"""
class DataCommonsDockerCommandLineJob(DataCommonsCommandLineJob):
    container_fmt_string = "docker run --rm -v /renci/irods:/renci/irods {}"

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
            self.container_fmt_string.format(str(image_tag))

        self._execute([], env, rm_tmpdir=rm_tmpdir, move_outputs=move_outputs)


"""
Make a tool object from a loaded cwl workflow/tool object
"""
def makeDataCommonsTool(cwl_obj, **kwargs):
    if not isinstance(cwl_obj, dict):
        raise WorkflowException("CWL object not a dict {}".format(cwl_obj))
    if "class" not in cwl_obj:
        raise WorkflowException("CWL object missing required class field")
    if cwl_obj.get("class") == "CommandLineTool":
        return DataCommonsCommandLineTool(cwl_obj, **kwargs)
    elif cwl_obj.get("class") == "ExpressionTool":
        return cwltool.draft2tool.ExpressionTool(cwl_obj, **kwargs)
    elif cwl_obj.get("class") == "Workflow":
        return cwltool.workflow.Workflow(cwl_obj, **kwargs)
    else:
        raise WorkflowException("Unsupported CWL class type : {}".format(cwl_obj.get("class")))


"""
Subclass of the cwltool CommandLineTool to override path mapping
"""
class DataCommonsCommandLineTool(cwltool.draft2tool.CommandLineTool):
    jobname_date_fmt = "%Y%m%d_%H%M%S.%f"

    def makeJobRunner(self, **kwargs):
        dockerReq, _ = self.get_requirement("DockerRequirement")
        # don't support the forced --use-container flag, or --no-container
        # if DockerRequirement is specified, always use it
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
        datestring = datetime.datetime.now().strftime(self.jobname_date_fmt)
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
        j.outdir = kwargs.get("outdir", None)
        j.basedir = kwargs.get("basedir", None)

        builder.pathmapper = None
        make_path_mapper_kwargs = kwargs
        if "stagedir" in make_path_mapper_kwargs:
            make_path_mapper_kwargs = make_path_mapper_kwargs.copy()
            del make_path_mapper_kwargs["stagedir"]

        # possibly remove
        #builder.pathmapper = self.makePathMapper(reffiles, builder.stagedir, **make_path_mapper_kwargs)
        builder.requirements = j.requirements

        #print(u"[job {}] command line bindings is {}".format(j.name, json.dumps(builder.bindings, indent=4)))

        def locToPath(p):
            if "path" not in p and "location" in p:
                p["path"] = uri_file_path(p["location"])
                #del p["location"]
        # change "Location" field on file class to "Path"
        cwltool.pathmapper.visit_class(builder.bindings, ("File","Directory"), locToPath)

        # pass tool stream definitions to job
        # evaluate each to a real path, then update path to be relative to basedir
        if self.tool.get("stdin"):
            orig_stdin = builder.do_eval(self.tool.get("stdin"))
            j.stdin = os.path.join(j.basedir, orig_stdin)
            _logger.debug("updated stdin from '{}' to '{}'".format(orig_stdin, j.stdin))
        if self.tool.get("stderr"):
            orig_stderr = builder.do_eval(self.tool.get("stderr"))
            j.stderr = os.path.join(j.basedir, orig_stderr)
            _logger.debug("updated stderr from '{}' to '{}'".format(orig_stderr, j.stderr))
        if self.tool.get("stdout"):
            orig_stdout = builder.do_eval(self.tool.get("stdout"))
            j.stdout = os.path.join(j.basedir, orig_stdout)
            _logger.debug("updated stdout from '{}' to '{}'".format(orig_stdout, j.stdout))

        j.command_line = flatten(list(map(builder.generate_arg, builder.bindings)))

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
        #_logger.debug("collect_output_ports finished: {}".format(ret))
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
                    #_logger.debug("gb evaluated to: '{}'".format(gb))
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

        return r


def get_chronos_client():
    return get_stars_client().scheduler.client


def get_stars_client():
    # store client in static function var, so there's only one instance
    if not hasattr(get_stars_client, "stars_client"):
        services_endpoint = os.getenv("DATACOMMONS_SERVICES_ENDPOINT")
        chronos_endpoint = os.getenv("DATACOMMONS_CHRONOS_ENDPOINT")
        chronos_proto = os.getenv("DATACOMMONS_CHRONOS_PROTO")
        if chronos_endpoint is None or chronos_proto is None:
            raise RuntimeError(
                "The datacommons module requires environment variable "
                "'DATACOMMONS_CHRONOS_ENDPOINT' and 'DATACOMMONS_CHRONOS_PROTO' to be set.")

        get_stars_client.stars_client = Stars(
            services_endpoints  = services_endpoint.split(",") if services_endpoint else None,
            scheduler_endpoints = chronos_endpoint.split(",") if chronos_endpoint else None)

        # override chronos client to use proto set in DATACOMMONS_CHRONOS_PROTO
        # TODO update stars to allow proto specification
        # https://github.com/stevencox/cwltool/issues/14
        get_stars_client.stars_client.scheduler.client = chronos.connect(chronos_endpoint, proto=chronos_proto)

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
        _logger.debug("compressing fields of job: {}".format(j.name))
        if isinstance(j, cwltool.draft2tool.ExpressionTool.ExpressionJob):
            # expression jobs run locally, no corresponding chronos job for it
            continue
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
            #_logger.debug("step in: {}".format(inp))
            if "valueFrom" in inp and "source" not in inp:
                _logger.error("Does not currently support valueFrom. Use source")
                return
            # resource urls for the input field and the field it gets it value from
            id = inp["id"]
            source = inp["source"]
            # trailing hash fragment in the resource url is the simple id
            id = id[id.rfind("#"):]
            _logger.debug("input field id: " + id)
            if isinstance(source, six.string_types):
                # single source field value
                source = source[source.rfind("#"):]
                new_j["in"].append(source)
            elif isinstance(source, list):
                # using MultipleInputFeatureRequirement
                # and this field has multiple input source links
                # supports linkMerge: merge_flattened or linkMerge: merge_nested
                # TODO have tested merge_flattened, need to test merge_nested
                for elem in source:
                    if isinstance(elem, list):      # merge nested
                        for nested_elem in elem:
                            nested_elem = nested_elem[nested_elem.rfind("#"):]
                            new_j["in"].append(nested_elem)
                    else:                           # merge flattened
                        elem = elem[elem.rfind("#"):]
                        new_j["in"].append(elem)

            #new_s["id"] = id

        new_j["out"] = []
        for outp in j_outp:
            #_logger.debug("step out: {}".format(outp))
            if isinstance(outp, six.string_types):
                id = outp[outp.rfind("#"):]
            elif "id" in outp:
                id = outp["id"]
                id = id[id.rfind("#"):]
            else:
                print("Out field for step is misformatted")
                return
            _logger.debug("output field id: " + id)
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
    _logger.debug("dependencies determined\n")
    #update_dependencies_in_chronos(jobs)
    update_dependencies_in_cache(jobs)


"""
Update a job in chronos with a schedule to run one time starting immediately
"""
def run_job_now(jobname):
    chronos_client = get_chronos_client()
    chronos_job = chronos_client.search(name=jobname)[0]
    if "parents" in chronos_job:
        _logger.warn("Cannot update job '{}' to run now, is dependent job".format(jobname))
    chronos_job["schedule"] = "R1//P1Y"
    _logger.debug("Setting job '{}' to run now.".format(jobname))
    chronos_client.update(chronos_job)


"""
Take a list of dict objects. Each dict is in the format:
{'name': <jobname>, 'parents': [<parentjobname>,...]}
For each element, attempt to update a job with given name, by adding parent jobs to it.
"""
def update_dependencies_in_cache(job_list):
    #chronos_client = get_chronos_client()
    for j in job_list:
        # get first job in cache with this name
        #chronos_job = chronos_client.search(name=j["name"])[0]
        cache_job = [c for c in job_cache if c["name"] == j["name"]]
        if len(cache_job) == 0:
            raise RuntimeError("Job '{}' not found in local job list".format(j["name"]))
        elif len(cache_job) > 1:
            _logger.warn("Multiple local jobs found with name '{}'".format(j["name"]))
        cache_job = cache_job[0]

        if "parents" not in cache_job:
            cache_job["parents"] = []

        if len(j["parents"]) == 0:
            # is a root level job, is dependent on nothing
            # set to run immediately
            cache_job["schedule"] = "R1//P1Y" #3rd segment doesn't matter with R1
        else:
            # is a child job, set parents, and set no schedule
            for p in j["parents"]:
                if p not in cache_job["parents"]:
                    _logger.debug("Adding {} as parent of {}".format(p, j["name"]))
                    cache_job["parents"].append(p)
            # job has parents, so cannot have schedule field
            del cache_job["schedule"]

        _logger.debug("Updated job '{}' in cache with:\n{}".format(j["name"], cache_job))
        #chronos_client.update(chronos_job)


"""
Take a list of dict objects. Each dict is in the format:
{'name': <jobname>, 'parents': [<parentjobname>,...]}
For each element, attempt to update a job with given name, by adding parent jobs to it.
"""
def update_dependencies_in_chronos(job_list):
    chronos_client = get_chronos_client()
    for j in job_list:
        # get first job in chronos with this name
        chronos_job = chronos_client.search(name=j["name"])[0]
        if len(j["parents"]) == 0:
            # is a root level job, is dependent on nothing
            # set to run immediately
            chronos_job["schedule"] = "R1//P1Y" #3rd segment doesn't matter with R1
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
# example iso8601 format: 2017-12-06T21:01:02.773Z
"""
def get_next_chronos_run(schedule_str):
    _logger.debug("schedule_str: {}".format(schedule_str))
    if not schedule_str:
        return None

    (repeat, start, interval) = schedule_str.split("/")
    now = datetime.datetime.now(pytz.utc)
    _logger.debug("now: {}".format(now))
    if start:
        start = dateutil.parser.parse(start)
        if interval:
            interval = isodate.parse_duration(interval)
            _logger.info("interval: {}".format(interval))
            elapsed_since_start = now - start

            _logger.info("elapsed_since_start: {}".format(elapsed_since_start))
            #print(elapsed_since_start)
            if elapsed_since_start.total_seconds() < 0:
                # hasn't started yet
                next_time = start
            else:
                # already started, calculate next
                # number of job intervals that have passed since the job was started

                interval_count = elapsed_since_start.total_seconds() / interval.total_seconds()
                #_logger.info("interval_count: {}".format(interval_count))

                # we want elapsed_since_start % interval
                _logger.debug("elapsed_seconds: {}".format(elapsed_since_start.total_seconds()))
                _logger.debug("interval_seconds: {}".format(interval.total_seconds()))
                mod = elapsed_since_start.total_seconds() % interval.total_seconds()
                _logger.debug("mod: {}".format(mod))

                time_to_next = datetime.timedelta(seconds=mod)

                """
                frac = interval_count % 1

                time_to_next = datetime.timedelta(seconds=frac * interval.total_seconds())
                """
                next_time = now + time_to_next

    else:
        # created with no start time, defaulting to now (will be filled in by chronos later)
        next_time = now

    # localize time
    local_tz = pytz.timezone(time.strftime("%Z"))
    next_time = next_time.replace(tzinfo=pytz.utc).astimezone(tz=local_tz)
    return next_time


"""
Look at job dependencies and include that in a table printout of the next [count]
jobs in chronos
#TODO implement count limiting. currently prints all jobs in chronos
#TODO it doesn't appear to be automatically sorting by next scheduled run or by creation time
"""
def show_upcoming_jobs_tree(count=20):
    chronos_client = get_chronos_client()
    jobs = chronos_client.list()
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
            d[name]["next_time"] = isoformat(next_time) if next_time else ""
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
    chronos_client = get_chronos_client()
    jobs = chronos_client.list()

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


"""
Verify that the job was created in a scheduler for a given stars client.
Log the endpoints, and the job schedule.
Return true if verified to exist, false otherwise.
"""
def verify_endpoint_job(jobname):
    # TODO update stars package to allow chronos /search functionality
    # so it is not returning all jobs into local memory
    chronos_client = get_chronos_client()
    jobs = chronos_client.list()
    found = False
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
                start = dateutil.parser.parse(start)
            else:
                start = "[immediately]"

            if interval:
                interval = isodate.parse_duration(interval)

            found = True
            _logger.info("Job '{}' created at endpoint(s). Repeating {} times, every {}, starting at {}"
                .format(jobname, repeat, interval, start))
    return found


"""
Post jobs to chronos in the order of the dependency hierarchy.
No job will be posted before its parent jobs.
"""
def post_chronos_jobs():
    global job_cache
    stars_client = get_stars_client()
    posted_names = []
    while len(job_cache) > 0:
        for j in job_cache:
            if "parents" not in j or len(j["parents"]) == 0:
                # not dependent on any other jobs, add now
                resp = stars_client.scheduler.add_job(**j)
                posted_names.append(j["name"])
                continue
            else:
                # has parents listed, check to see if all parents have been posted first
                all_parents_posted = True
                for p in j["parents"]:
                    if p not in posted_names:
                        all_parents_posted = False
                if all_parents_posted:
                    resp = stars_client.scheduler.add_job(**j)
                    posted_names.append(j["name"])
                    continue

        # remove those posted in this pass from the cache
        job_cache = [jc for jc in job_cache if jc["name"] not in posted_names]



"""
Send the commands to the data commons API.
Creates the jobs to initially run 100 years in future, and every year after.
Start time must be reduced later via api in order for them to run before that.
"""
def _datacommons_popen(
        jobname, # type: Text
        commands,  # type: List[Text]
        env,  # type: Union[MutableMapping[Text, Text], MutableMapping[str, str]]
        cwd,  # type: Text
        container_command=None, # type string
        stdin=None, # type: Text (file path)
        stdout=None, # type: Text (file path)
        stderr=None, # type: Text (file path)
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

    schedule = "R//P1Y"

    kwargs = {
        "name": jobname,
        "command": command,
        "owner": "ted@job.org",
        "schedule": schedule,
        "execute_now": False,
        "shell": True
    }

    #if container_args:
        # add container arguments to the kwargs
    #    kwargs["container"] = container_args

    #pivot.scheduler.add_job(
    #    **kwargs
    #)

    #if verify_endpoint_job(pivot, jobname):
    #    rcode = 0
    #else:
    #    rcode = 1

    job_cache.append(kwargs)
    _logger.debug("job '{}' added to local job cache".format(kwargs["name"]))
    rcode = 0
    return rcode
