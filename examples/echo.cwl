cwlVersion: v1.0
class: CommandLineTool
baseCommand: echo
inputs:
  message:
    type: string
    inputBinding:
      position: 1
  output_location:
    type: string
outputs:
  - id: echo_output
    type: stdout
stdout: $(inputs.output_location)
