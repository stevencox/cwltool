from __future__ import absolute_import
import unittest
import pytest
from mock import mock, patch
import sys
import os
import chronos
import logging

import datetime
import pytz
import isodate
import time

from cwltool.main import main
from cwltool import data_commons
isoformat = data_commons.isoformat

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

data_commons._logger.setLevel(logging.DEBUG)

class TestDataCommons(unittest.TestCase):


    def test_datacommons_popen_no_container(self):
        os.environ["DATACOMMONS_CHRONOS_ENDPOINT"] = "fakeendpoint.net/chronos"

        with patch.object(data_commons, "verify_endpoint_job") as mock_verify_endpoint_job, \
             patch.object(chronos.ChronosClient, "add") as mock_chronos_add:

            data_commons._datacommons_popen(
                jobname="testjob",
                commands=["echo", "hello", "world"],
                env=None,
                cwd="/testcwd",
                container_command=None,
                stdin="/testcwd/echo.stdin",
                stdout="/testcwd/echo.stdout",
                stderr="/testcwd/echo.stderr"
            )
            self.assertTrue(mock_chronos_add.called)

            args, kwargs = mock_chronos_add.call_args
            self.assertIsNotNone(args)
            self.assertEqual(len(args), 1)
            job = args[0]

            self.assertEqual(job["name"], "testjob")
            self.assertEqual(job["command"],
                    "echo hello world < /testcwd/echo.stdin > " \
                    + "/testcwd/echo.stdout 2> /testcwd/echo.stderr")
            self.assertEqual(job["owner"], "ted@job.org")
            self.assertEqual(job["runAsUser"], "evryscope")
            self.assertTrue("R/" in job["schedule"] and "/P1Y" in job["schedule"])
            self.assertTrue(job["shell"])


    def test_datacommons_popen_with_container(self):
        os.environ["DATACOMMONS_CHRONOS_ENDPOINT"] = "fakeendpoint.net/chronos"

        with patch.object(data_commons, "verify_endpoint_job") as mock_verify_endpoint_job, \
             patch.object(chronos.ChronosClient, "add") as mock_chronos_add:

            data_commons._datacommons_popen(
                jobname="testjob",
                commands=["echo", "hello", "world"],
                env=None,
                cwd="/testcwd",
                container_command=
                    data_commons.DataCommonsDockerCommandLineJob.container_fmt_string \
                                  .format("centos:centos7"),
                stdin="/testcwd/echo.stdin",
                stdout="/testcwd/echo.stdout",
                stderr="/testcwd/echo.stderr"
            )
            self.assertTrue(mock_chronos_add.called)

            args, kwargs = mock_chronos_add.call_args
            self.assertIsNotNone(args)
            self.assertEqual(len(args), 1)
            job = args[0]

            self.assertEqual(job["name"], "testjob")
            self.assertEqual(job["command"],
                    "docker run --rm -v /renci/irods:/renci/irods centos:centos7 " \
                    + "echo hello world < /testcwd/echo.stdin > " \
                    + "/testcwd/echo.stdout 2> /testcwd/echo.stderr")
            self.assertEqual(job["owner"], "ted@job.org")
            self.assertEqual(job["runAsUser"], "evryscope")
            self.assertTrue("R/" in job["schedule"] and "/P1Y" in job["schedule"])
            self.assertTrue(job["shell"])


    def test_verify_endpoint_job(self):
        os.environ["DATACOMMONS_CHRONOS_ENDPOINT"] = "fakeendpoint.net/chronos"
        list_ret = [
            {'name':'job1',
                'command':'ls',
                'owner':'owner@test.net',
                'runAsUser':'fakeuser',
                'schedule':'R1/9999-01-01T12:00:00.000Z/P1D',
                'shell':True},
            {'name':'job2',
                'command':'ls',
                'owner':'owner@test.net',
                'runAsUser':'fakeuser',
                'schedule':'R1/9999-02-02T12:00:00.000Z/P1D',
                'shell':True},
            {'name':'job3',
                'command':'ls',
                'owner':'owner@test.net',
                'runAsUser':'fakeuser',
                'schedule':'R1/9999-03-03T12:00:00.000Z/P1D',
                'shell':True}
        ]

        with patch.object(chronos.ChronosClient, "list", return_value=list_ret) as mock_chronos_list:

            log_monitor = LogMonitor()
            data_commons._logger.addHandler(log_monitor)
            verified = data_commons.verify_endpoint_job(data_commons.get_stars_client(), "job1")
            self.assertTrue(verified)
            verified = data_commons.verify_endpoint_job(data_commons.get_stars_client(), "job2")
            self.assertTrue(verified)
            verified = data_commons.verify_endpoint_job(data_commons.get_stars_client(), "job3")
            self.assertTrue(verified)

            print(log_monitor.log_list)
            self.assertEqual(len(log_monitor.log_list), 3)
            self.assertTrue("job1" in log_monitor.log_list[0])
            self.assertTrue("job2" in log_monitor.log_list[1])
            self.assertTrue("job3" in log_monitor.log_list[2])


    def test_cwltool_main_datacommons_hello_workflow(self):
        os.environ["DATACOMMONS_CHRONOS_ENDPOINT"] = "fakeendpoint.net/chronos"
        with patch.object(data_commons, "_datacommons_popen", return_value=0) \
                as mock_popen, \
             patch.object(data_commons, "set_job_dependencies") \
                as mock_set_job_dependencies, \
             patch.object(data_commons, "show_upcoming_jobs_tree", return_value="") \
                as mock_show_upcoming:

            main(["--data-commons", "tests/wf/hello-workflow.cwl", "--usermessage", "hello"])


            popen_calls = mock_popen.call_args_list
            self.assertEqual(len(popen_calls), 1)
            for args, kwargs in popen_calls:
                print(kwargs)
                jobname_elems = kwargs["jobname"].split("-")
                self.assertEqual("datacommonscwl", jobname_elems[0])
                self.assertTrue(isinstance( # make sure the 2nd elem in jobname is a date
                    datetime.datetime.strptime(
                        jobname_elems[1],
                        data_commons.DataCommonsCommandLineTool.jobname_date_fmt),
                    datetime.datetime))
                self.assertEqual(kwargs["commands"], ["echo", "-n", "-e", "hello"])

    def test_set_job_dependencies(self):
        jobs = []
        with patch.object(data_commons, "_datacommons_popen", return_value=0) \
                as mock_popen, \
             patch.object(data_commons, "update_dependencies_in_chronos") \
                as mock_update_dependencies, \
             patch.object(data_commons, "show_upcoming_jobs_tree", return_value="") \
                as mock_show_upcoming:

            main(["--data-commons", "tests/data_commons/functional-wf.cwl"])

            #mock_update_dependencies.assert_called()
            self.assertEqual(len(mock_update_dependencies.call_args_list), 1)
            args, kwargs = mock_update_dependencies.call_args_list[0]
            job_list = args[0]
            print("job_list: {}".format(job_list))
            self.assertEqual(len(job_list), 8) # 8 jobs created by workflow

            # check jobs for correct in/out fields, and parent fields if applicable
            # verify jobs for echo_w step
            echo_w_jobs = [j for j in job_list if "#echo_w/txt" in j["out"]]
            self.assertEqual(len(echo_w_jobs), 1)
            echo_w_job = echo_w_jobs[0]
            self.assertEqual(echo_w_job["in"], ["#letters0"])
            self.assertEqual(len(echo_w_job["parents"]), 0)

            # verify jobs for echo_x step
            echo_x_jobs = [j for j in job_list if "#echo_x/txt" in j["out"]]
            self.assertEqual(len(echo_x_jobs), 2)
            for echo_x in echo_x_jobs:
                self.assertEqual(echo_x["in"], ["#letters1", "#letters2"])
                self.assertEqual(len(echo_x["parents"]), 0)


            # verify jobs for echo_y step
            echo_y_jobs = [j for j in job_list if "#echo_y/txt" in j["out"]]
            self.assertEqual(len(echo_y_jobs), 3)
            for echo_y in echo_y_jobs:
                self.assertEqual(echo_y["in"], ["#letters3", "#letters4"])
                self.assertEqual(len(echo_y["parents"]), 0)

            # verify jobs for echo_z step
            echo_z_jobs = [j for j in job_list if "#echo_z/txt" in j["out"]]
            self.assertEqual(len(echo_z_jobs), 1)
            echo_z_job = echo_z_jobs[0]
            self.assertEqual(echo_z_job["in"], ["#letters5"])
            self.assertEqual(len(echo_z_job["parents"]), 0)

            # verify jobs for cat step
            cat_jobs = [j for j in job_list if "#cat/txt" in j["out"]]
            self.assertEqual(len(cat_jobs), 1)
            cat_job = cat_jobs[0]
            self.assertEqual(
                cat_job["in"],
                ["#echo_w/txt", "#echo_x/txt", "#echo_y/txt", "#echo_z/txt"])
            self.assertEqual(len(cat_job["parents"]), 7)

    def test_get_next_chronos_run(self):
        now = datetime.datetime.now(pytz.utc)
        #start_str = isodate.datetime_isoformat(now)
        #schedule_str = "R/{}/PT1H".format(start_str)

        local_tz = pytz.timezone(time.strftime("%Z"))
        local_now = now.replace(tzinfo=pytz.utc).astimezone(tz=local_tz)
        _logger.debug("local_now: {}".format(local_now))
        # test data
        datetime_1 = now + datetime.timedelta(hours=-1, minutes=-12) # scheduled start time
        schedule_1 = "R/{}/PT1H".format(isoformat(datetime_1)) # schedule with interval
        delta_1 = datetime.timedelta(minutes=12) # expected delta to next run

        datetime_2 = now + datetime.timedelta(hours=-12, minutes=-25)
        schedule_2 = "R/{}/PT5H".format(isoformat(datetime_2))
        delta_2 = datetime.timedelta(hours=2, minutes=25)

        datetime_3 = now + datetime.timedelta(days=-7, hours=-1, minutes=-6)
        schedule_3 = "R/{}/P2DT15M".format(isoformat(datetime_3))
        delta_3 = datetime.timedelta(days=1, minutes=21)

        datetime_4 = now + datetime.timedelta(days=-1, hours=-3, minutes=-5)
        schedule_4 = "R/{}/PT1H".format(isoformat(datetime_4))
        delta_4 = datetime.timedelta(minutes=5)

        l = [
            (datetime_1, schedule_1, delta_1),
            (datetime_2, schedule_2, delta_2),
            (datetime_3, schedule_3, delta_3),
            (datetime_4, schedule_4, delta_4),
        ]
        try:
            class FakeDatetime(datetime.datetime):
                @classmethod
                def now(cls, tz=None):
                    return now # now in UTC

            dt_bkp = datetime.datetime
            datetime.datetime = FakeDatetime
            for idx, (dt, schedule, delta) in enumerate(l):
                _logger.debug("test set: {}, timedelta: {}".format(idx+1, delta))
                #with patch.object(datetime.datetime.now, return_value=now) as mock_now:
                next_time = data_commons.get_next_chronos_run(schedule)
                self.assertEqual(next_time, local_now + delta)
        except AssertionError as e:
            _logger.error("calculated next run differed from expected next run by: {}".format(
                next_time - (local_now + delta)
            ))
            raise e
        finally:
            datetime.datetime = dt_bkp

        self.assertTrue(False)



class LogMonitor(logging.StreamHandler):
    def __init__(self):
        super(LogMonitor, self).__init__()
        self.log_list = []

    """
    override
    """
    def emit(self, record):
        self.log_list.append(self.format(record))
    def flush(self, record):
        pass
    def reset(self):
        self.log_list = []
