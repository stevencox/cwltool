cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["wget"]
inputs:
  - id: options
    type: string
    inputBinding:
      position: 1
  - id: directory
    type: string
    inputBinding:
      prefix: -P
      position: 2
  - id: url
    type: string
    inputBinding:
      position: 3

outputs:
  - id: download_basename
    type: File
    outputBinding:
      glob: $(inputs.url.split("/")[inputs.url.split("/").length-1])
requirements:
  - class: InlineJavascriptRequirement
