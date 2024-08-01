const datasetContainer = $("#dataset");
$(document).ready(async function () {
  datasetContainer.empty();
  cloud
    .add(origin + "/api/dataset", {
      name: "dataset",
      callback: (data) => {},
    })
    .then((dataset) => {
      $.each(dataset, function (kelas, list) {
        datasetContainer.append(`<p class="title">${capEachWord(kelas.replace("_", " "))}</p>`);
        const imageContainer = $(`<div class="image-container"></div>`);
        $.each(list, function (i, image) {
          imageContainer.append(`<img src="static/img/dataset/${image}">`);
        });
        datasetContainer.append(imageContainer);
      });
    });
});
