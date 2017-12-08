cwlVersion: v1.0
class: Workflow
id: complex-workflow
inputs:
  archive:
    type: string
  files:
    type:
      type: array
      items: File
  new_archive:
    type: string
  touch_files:
    type:
      type: array
      items: string
  echo_message:
    type: string
  echo_output_location:
    type: string

outputs:
  output_archive:
    type: File
    outputSource: re_tar_step/archive_out

steps:
  tar_step:
    run: tar.cwl
    in:
      archive_file:
        source: "#archive"
      file_list:
        source: "#files"
    out: [archive_out]

  touch_step:
    run: touch.cwl
    scatter: filename
    scatterMethod: dotproduct
    in:
      waitfor:
        source: "#tar_step/archive_out"
      filename:
        source: "#touch_files"
    out: [file_out]

  echo_step:
    run: echo.cwl
    in:
      message: "#echo_message"
      output_location: "#echo_output_location"
    out: [echo_output]

  re_tar_step:
    run: tar.cwl
    in:
      archive_file:
        source: "#archive"
      file_list:
        source: ["#touch_step/file_out", "#files", "#echo_step/echo_output"]
        linkMerge: merge_flattened
    out: [archive_out]

  rm_step:
    run: rm.cwl
    scatter: filename
    scatterMethod: dotproduct
    in:
      waitfor:
        source: "#re_tar_step/archive_out"
      filename:
        source: ["#files", "#touch_step/file_out", "#echo_step/echo_output"]
        linkMerge: merge_flattened
    out: [standard_out]

requirements:
  - class: StepInputExpressionRequirement
  - class: ScatterFeatureRequirement
  - class: MultipleInputFeatureRequirement
