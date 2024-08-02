$("body").on("submit", "#form-prediksi", function (e) {
  e.preventDefault();
  const loader = $("#loader-prediksi");
  const form = $(this);
  const data = new FormData();
  data.append("name", $("#name").val());
  data.append("image", $("#image")[0].files[0]);

  form.find("input, button, textarea").prop("disabled", true);
  loader.fadeIn("fast");
  $.ajax({
    type: "POST",
    url: origin + "/api/prediksi",
    data: data,
    contentType: false,
    processData: false,
    cache: false,
    success: function (data) {
      loader.fadeOut("fast");
      form.find("input, button, textarea").prop("disabled", false);
      $("#hasil").text(data.label);
      $.each(data.classes, function (i, c) { 
        $("#nilai-" + c).text(data.predictions[i]);
      });
      $("#form-prediksi").trigger("reset");
      M.toast({ html: "Data diprediksi" });
    },
  });
});

$(document).ready(async function () {
  cloud
    .add(origin + "/api/dataset", {
      name: "dataset",
      callback: (data) => {},
    })
    .then((dataset) => {
      $.each(dataset, function (kelas, images) {
        $("#table-prediksi tbody").append("<tr><td>" + kelas + `</td><td id="nilai-${kelas}"></td></tr>`);
      });
    });
  cloud
    .add(origin + "/api/models", {
      name: "models",
      callback: (data) => {},
    })
    .then((models) => {
      $.each(models, function (i, model) {
        $("#name").append("<option value='" + model.name + "'>" + capEachWord(model.name.replace("_", " ")) + "</option>");
      });
      $("#name").formSelect();
    });
});
