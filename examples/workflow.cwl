cwlVersion: v1.0
class: Workflow
id: workflow
inputs:
  archive:
    type: string
  files:
    type:
      type: array
      items: File
  new_file:
    type: string

outputs:
  output_archive:
    type: File
    outputBinding:
      glob: "#rename/renamed_file"

steps:
  tar_step:
    run: tar.cwl
    in:
      archive_file:
        source: "#archive"
      file_list:
        source: "#files"
    out: [archive_out, standard_out]

  rename_step:
    run: rename.cwl
    in:
      original_name:
        source: tar_step/archive_out
      new_name:
        source: "#new_file"
    out: [renamed_file]
