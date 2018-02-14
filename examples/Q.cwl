class: CommandLineTool
cwlVersion: v1.0
baseCommand:
  - Q
inputs:
  - id: t
    type: File
    inputBinding:
      position: 1
      prefix: -t
  - id: c
    type: File
    inputBinding:
      position: 2
      prefix: -c
  - id: o
    type: string
    inputBinding:
      position: 3
      prefix: -o
  - id: options
    type: string
    inputBinding:
      position: 4
outputs:
  - id: bed_file
    type: File
    outputBinding:
      glob: "*-narrowpeak.bed"
  - id: tab_file
    type: File
    outputBinding:
      glob: "*-quality-statistics.tab"
stdout: Q.stdout
requirements:
  - class: InlineJavascriptRequirement
