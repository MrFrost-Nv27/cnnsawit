const table = {
  pelatihan: $("#table-pelatihan").DataTable({
    responsive: true,
    ajax: {
      url: origin + "/api/pelatihan",
      dataSrc: "data",
    },
    columns: [
      {
        title: "#",
        data: "id",
        render: function (data, type, row, meta) {
          return meta.row + meta.settings._iDisplayStart + 1;
        },
      },
      { title: "Nama Model", data: "nama" },
      { title: "Algoritma", data: "algoritma", render: function (data) {
        return data == "nb" ? "Naive Bayes" : "C 4.5";
      }},
      { title: "K Fold", data: "kfold" },
      { title: "Akurasi", data: "akurasi" },
      {
        title: "Aksi",
        data: "id",
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

  $.ajax({
    type: "POST",
    url: origin + "/api/pelatihan",
    data: $(this).serialize(),
    success: function (data) {
      table.pelatihan.ajax.reload();
      $("#form-pelatihan").trigger("reset");
      M.toast({ html: "Data baru berhasil ditambahkan" });
    },
  });
});

$(document).ready(async function () {});
