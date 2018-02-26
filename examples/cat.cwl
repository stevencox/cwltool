class: CommandLineTool
cwlVersion: v1.0
baseCommand:
  - cat
inputs:
  - id: file
    type: File
    inputBinding:
      position: 1
  - id: output_file
    type: string
outputs:
  - id: standard_out
    type: stdout
    outputBinding: {}
stdout: $(inputs.output_file)
