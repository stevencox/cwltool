cwlVersion: v1.0
class: ExpressionTool
requirements:
  - class: InlineJavascriptRequirement
inputs: []
outputs:
  - id: date
    type: string
  - id: time
    type: string
  - id: datetime
    type: string
expression: >
  ${
    var d = new Date();
    var datetime = d.toISOString();
    var date = datetime.split("T")[0];
    var time = datetime.split("T")[1];
    return {"date":date, "time":time, "datetime":datetime};
  }
