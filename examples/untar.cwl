cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["tar", "xf"]
inputs:
  - id: archive_file
    type: string
    inputBinding:
      position: 1
outputs:
  - id: standard_out
    type: stdout
  - id: files_out
    type:
      type: array
      items: File
    outputBinding:
      glob: "*.txt"

stdout: stdout
