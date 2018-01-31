cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["rm"]
inputs:
  - id: filename
    type: File
    inputBinding:
      position: 1
outputs:
  - id: standard_out
    type: stdout
stdout: $(inputs.filename.location.replace("file://","") + ".rm.cwl.stdout")
requirements:
  - class: InlineJavascriptRequirement
