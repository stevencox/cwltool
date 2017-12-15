cwlVersion: v1.0
class: Workflow
id: simple-workflow
inputs: []

outputs:
  datetime:
    type: string
    outputSource: date_step/datetime

steps:
  date_step:
    run: getdate.cwl
    in: []
    out: [datetime]

requirements: []
