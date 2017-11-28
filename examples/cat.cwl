cwlVersion: v1.0
class: CommandLineTool
baseCommand: cat
inputs:
  - id: file
    type: File
    inputBinding:
      position: 1
  - id: output_file
    type: string
outputs:
  standard_out:
    type: stdout
stdout: $(inputs.output_file)
