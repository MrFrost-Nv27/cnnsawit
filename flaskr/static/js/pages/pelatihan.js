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
      { title: "Nama Model", data: "name", render: (data) => capEachWord(data.replace("_", " ")) },
      { title: "Akurasi", data: "akurasi" },
      {
        title: "Aksi",
        data: "name",
        render: (data, type, row) => {
          return `<div class="table-control">
            <a role="button" class="btn waves-effect waves-light btn-action red" data-action="delete" data-id="${data}"><i class="material-icons">delete</i></a>
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
});
