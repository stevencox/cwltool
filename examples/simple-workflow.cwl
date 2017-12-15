cwlVersion: v1.0
class: Workflow
id: simple-workflow
inputs:
  echo_step_output_location:
    type: string

outputs:
  wf_output:
    type: File
    outputSource: echo_step/echo_output

steps:
  date_step:
    run: getdate.cwl
    in: []
    out: [datetime]
  echo_step:
    run: echo.cwl
    in:
      message:
        source: "#date_step/datetime"
      output_location:
        source: "#echo_step_output_location"
    out: [echo_output]

requirements: []
