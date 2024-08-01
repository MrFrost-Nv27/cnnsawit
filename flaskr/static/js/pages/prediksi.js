const table = {
  periksa: $("#table-periksa").DataTable({
    responsive: true,
    ajax: {
      url: origin + "/api/periksa",
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
      { title: "Nama", data: "nama" },
      { title: "Usia", data: "usia" },
      { title: "Jenis Kelamin", data: "jenis_kelamin" },
      { title: "Berat Badan", data: "bb" },
      { title: "Tinggi Badan", data: "tb" },
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

$(document).ready(async function () {});
