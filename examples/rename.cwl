cwlVersion: v1.0
class: CommandLineTool
baseCommand: ["mv"]
inputs:
  - id: original_name
    type: File
    inputBinding:
      position: 1
  - id: new_name
    type: string
    inputBinding:
      position: 2
outputs:
  - id: renamed_file
    type: File
    outputBinding:
      glob: $(inputs.new_name)
