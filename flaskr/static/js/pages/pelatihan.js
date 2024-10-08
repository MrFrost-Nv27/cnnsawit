const table = {
  pelatihan: $("#table-pelatihan").DataTable({
    responsive: true,
    ajax: {
      url: origin + "/api/models",
      dataSrc: "",
    },
    columns: [
      {
        title: "#",
        data: "name",
        render: function (data, type, row, meta) {
          return meta.row + meta.settings._iDisplayStart + 1;
        },
      },
      {
        title: "Nama Model",
        data: "name",
        render: (data) => capEachWord(data.replace("_", " ")),
      },
      { title: "Akurasi", data: "akurasi" },
      {
        title: "Aksi",
        data: "name",
        render: (data, type, row) => {
          return `<div class="table-control">
            <a role="button" class="btn waves-effect waves-light btn-action red" data-action="delete" data-id="${data}"><i class="material-icons">delete</i></a>
            <a role="button" class="btn waves-effect waves-light btn-action blue" data-action="show-accuracy" data-id="${data}"><i class="material-icons">graphic_eq</i></a>
            </div>`;
        },
      },
    ],
  }),
};

$("body").on("submit", "#form-pelatihan", function (e) {
  e.preventDefault();
  const loader = $("#loader-pelatihan");
  const form = $(this);
  const data = $(this).serialize();
  form.find("input, button, textarea").prop("disabled", true);
  loader.fadeIn("fast");
  $.ajax({
    type: "POST",
    url: origin + "/api/pelatihan",
    data: data,
    success: function (data) {
      loader.fadeOut("fast");
      form.find("input, button, textarea").prop("disabled", false);
      table.pelatihan.ajax.reload();
      $("#form-pelatihan").trigger("reset");
      M.toast({ html: "Model baru berhasil dilatih" });
    },
  });
});

// Handle accuracy plot button click
$("body").on("click", ".btn-action", function () {
  const action = $(this).data("action");
  const id = $(this).data("id");

  if (action === "delete") {
    $.ajax({
      type: "DELETE",
      url: origin + "/api/models/" + id,
      success: function (data) {
        table.pelatihan.ajax.reload();
        M.toast({ html: "Model berhasil dihapus" });
      },
    });
  } else if (action === "show-accuracy") {
    const imageUrl = origin + "/api/models/" + id + "/accuracy_plot";
    $("#accuracyImage").attr("src", imageUrl);
    $("#accuracyModal").modal("open");
  }
});

$(document).ready(async function () {
  $(document).ready(async function () {
    cloud
      .add(origin + "/api/dataset", {
        name: "dataset",
        callback: (data) => {},
      })
      .then((dataset) => {});
    cloud
      .add(origin + "/api/models", {
        name: "models",
        callback: (data) => {},
      })
      .then((models) => {});
  });
  $(".modal").modal();
});
