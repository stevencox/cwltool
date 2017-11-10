import argparse
import sys
import os
import copy
import json
import datetime
from pprint import pprint
from typing import Any, Callable, Dict, List, Text, Union, cast

import cwltool.load_tool
import cwltool.resolver
import cwltool.draft2tool
import cwltool.main
import cwltool.process
from cwltool.flatten import flatten
from cwltool.errors import WorkflowException

from schema_salad.ref_resolver import uri_file_path
from schema_salad.sourceline import SourceLine, indent
from stars.stars import Stars

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
        # copied and pruned from cwltool.job.JobBase._execute
        shouldquote = lambda x: False

        print(u"[job %s] %s$ %s%s%s%s",
                     self.name,
                     self.outdir,
                     " \\\n    ".join([shellescape.quote(Text(arg)) if shouldquote(Text(arg)) else Text(arg) for arg in
                                       (runtime + self.command_line)]),
                     u' < %s' % self.stdin if self.stdin else '',
                     u' > %s' % os.path.join(self.outdir, self.stdout) if self.stdout else '',
                     u' 2> %s' % os.path.join(self.outdir, self.stderr) if self.stderr else '')

        outputs = {}  # type: Dict[Text,Text]

        try:
            commands = [Text(x) for x in (runtime + self.command_line)]
            print("Commands: " + str(commands))
            rcode = _datacommons_popen(
                self.name,
                commands,
                env=env,
                cwd=self.outdir
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
                    print(u"'%s' not found", runtime[0])
                else:
                    print(u"'%s' not found", self.command_line[0])
            else:
                print("Exception while running job")
            processStatus = "permanentFail"
        except WorkflowException as e:
            print(u"[job %s] Job error:\n%s" % (self.name, e))
            processStatus = "permanentFail"
        except Exception as e:
            print("Exception while running job: " + str(e))
            import traceback
            traceback.print_tb(e.__traceback__)
            processStatus = "permanentFail"

        if processStatus != "success":
            print(u"[job {}] completed {}".format(self.name, processStatus))
        else:
            print(u"[job {}] completed {}".format(self.name, processStatus))

        print(u"[job {}] {}".format(self.name, json.dumps(outputs, indent=4)))

        #kferriter Maybe want some sort of callback
        #self.output_callback(outputs, processStatus)


"""
Make a tool object from a loaded cwl workflow/tool object
"""
def makeDataCommonsTool(cwl_obj, **kwargs):
    # not a cwl object, so stop
    if not isinstance(cwl_obj, dict):
        raise WorkflowException("CWL object not a dict {}".format(cwl_obj))
    # These classes are what we want to wrap
    if cwl_obj.get("class") == "CommandLineTool":
        return DataCommonsCommandLineTool(cwl_obj, **kwargs)
    else:
        raise WorkflowException("Unsupported CWL class type : {}".format(cwl_obj.get("class")))


"""
Subclass of the cwltool CommandLineTool to override path mapping
"""
class DataCommonsCommandLineTool(cwltool.draft2tool.CommandLineTool):

    def makeJobRunner(self, **kwargs):
        return DataCommonsCommandLineJob()

    def makePathMapper(self, reffiles, stagedir, **kwargs):
        #return super().makePathMapper(reffiles, stagedir, **kwargs)
        pass

    def job(self, job_order, output_calbacks, **kwargs):
        # copied and pruned from cwltool.draft2tool.CommandLineTool.job
        #jobname = uniquename(kwargs.get("name", shortname(self.tool.get("id", "job"))))
        datestring = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
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

        builder.pathmapper = None
        make_path_mapper_kwargs = kwargs
        if "stagedir" in make_path_mapper_kwargs:
            make_path_mapper_kwargs = make_path_mapper_kwargs.copy()
            del make_path_mapper_kwargs["stagedir"]

        # need this?
        builder.pathmapper = self.makePathMapper(reffiles, builder.stagedir, **make_path_mapper_kwargs)
        builder.requirements = j.requirements

        # These if statements aren't really doing anything right now
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

        print(u"[job %s] command line bindings is %s", j.name, json.dumps(builder.bindings, indent=4))


        j.command_line = flatten(list(map(builder.generate_arg, builder.bindings)))

        j.pathmapper = builder.pathmapper

        yield j

class DataCommonsPathMapper(cwltool.pathmapper.PathMapper):
    def __init__(self, referenced_files, basedir):
        self._pathmap = {}
        self.stagedir = basedir
        super().setup(dedup(referenced_files), basedir)


"""
Send the commands to the data commons API
"""
def _datacommons_popen(
        jobname,
        commands,  # type: List[Text]
        env,  # type: Union[MutableMapping[Text, Text], MutableMapping[str, str]]
        cwd,  # type: Text
    ):

    #print ("||STARS||> cmd: %s in: %s out: %s err: %s env: %s cwd: %s" % (commands, stdin, stdout, stderr, env, cwd))
    print("STARS: cmd: {}".format(commands))
    pivot = Stars(
        services_endpoints  = [ "https://stars-app.renci.org/marathon"],
        scheduler_endpoints = [ "stars-app.renci.org/chronos" ])
    pivot.scheduler.add_job (
        name=jobname,
        command=' '.join (commands),
        owner="ted@job.org",
        runAsUser="evryscope",
        schedule="R/3000-01-01T00:00:00Z/PT60M",
        constraints=[
         [
             "hostname",
             "EQUALS",
             "stars-dw4.edc.renci.org"
         ]
        ],
        execute_now=True
    )

    rcode = 0
    return rcode
