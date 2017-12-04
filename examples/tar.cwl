cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["tar", "-cf", "-"]
inputs:
  - id: archive_file
    type: string
    inputBinding:
      position: 1
  - id: file_list
    type:
      type: array
      items: File
    inputBinding:
      position: 2
outputs:
  - id: archive_out
    type: File
    outputBinding:
      glob: $(inputs.archive_file)
stdout: $(inputs.archive_file)
