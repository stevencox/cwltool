cwlVersion: v1.0
class: Workflow
id: complex-workflow
inputs:
  wget_list:
    type:
      type: array
      items: string
  wget_directory:
    type: string
  wget_options:
    type: string
  Q_t:
    type: File
  Q_c:
    type: File
  Q_o:
    type: string
  Q_options:
    type: string

outputs: {}

steps:
  wget_step:
    run: wget.cwl
    scatter: url
    scatterMethod: dotproduct
    in:
      url:
        source: "#wget_list"
      directory:
        source: "#wget_directory"
      options:
        source: "#wget_options"
    out: [download_basename]
    requirements:
      - class: DockerRequirement
        dockerPull: heliumdatacommons/bio-tools

  Q_step:
    run: Q.cwl
    in:
      waitfor:
        source: "#wget_step/download_basename"
      t:
        source: "#Q_t"
      c:
        source: "#Q_c"
      o:
        source: "#Q_o"
      options:
        source: "#Q_options"
    out: [bed_file, tab_file]
    requirements:
      - class: DockerRequirement
        dockerPull: heliumdatacommons/bio-tools


requirements:
  - class: StepInputExpressionRequirement
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement
