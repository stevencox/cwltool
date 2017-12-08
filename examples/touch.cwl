cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["touch"]
inputs:
  - id: filename
    type: string
    inputBinding:
      position: 1
outputs:
  - id: file_out
    type: File
    outputBinding:
      glob: $(inputs.filename)
  - id: standard_out
    type: stdout
stdout: $(inputs.filename+".touch.cwl.stdout")
requirements:
  - class: InlineJavascriptRequirement
