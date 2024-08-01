class W {
  static wasInit = false;
  static config = {
    el: {
      provinsi: $("select[name=provinsi]"),
      kabupaten: $("select[name=kota]"),
      kecamatan: $("select[name=kecamatan]"),
      desa: $("select[name=desa]"),
    },
    desc: {
      provinsi: "Provinsi",
      kabupaten: "Kabupaten/Kota",
      kecamatan: "Kecamatan",
      desa: "Desa",
    },
    dataset: [],
  };
  static init(
    options = {
      dataset: [],
    }
  ) {
    if (this.wasInit) {
      throw new Error("Already initialized");
    }
    for (const [key, value] of Object.entries(options)) {
      if (typeof value === "object") {
        for (const [k, v] of Object.entries(value)) {
          this.config[key][k] = v;
        }
        continue;
      }
      this.config[key] = value;
    }

    this.reset();
    this.config.el.provinsi.on("change", this.#onProvinsi);
    this.config.el.kabupaten.on("change", this.#onKabupaten);
    this.config.el.kecamatan.on("change", this.#onKecamatan);
    this.wasInit = true;
    return this;
  }

  static setData(ds = []) {
    this.config.dataset = ds;
    return this;
  }

  static destroy() {
    this.config.el.provinsi.off("change", this.#onProvinsi);
    this.config.el.kabupaten.off("change", this.#onKabupaten);
    this.config.el.kecamatan.off("change", this.#onKecamatan);
    this.wasInit = false;
    this.config = {
      el: {
        provinsi: $("select[name=provinsi]"),
        kabupaten: $("select[name=kota]"),
        kecamatan: $("select[name=kecamatan]"),
        desa: $("select[name=desa]"),
      },
      dataset: [],
    };
  }

  static reset() {
    for (const [key, value] of Object.entries(this.config.el)) {
      value.empty().append(`<option value="" disabled selected>Pilih ${this.config.desc[key]}</option>`);
      value.formSelect();
    }
    if (this.config.dataset.length > 0) {
      this.config.dataset
        .filter((d) => d.kode.length === 2)
        .forEach((d) => {
          this.config.el.provinsi.append(`<option value="${d.kode}">${d.nama}</option>`);
        });
      this.config.el.provinsi.formSelect();
    }
    return this;
  }

  static #onProvinsi(e) {
    if ($(this).val()) {
      W.config.el.kecamatan.empty().append(`<option value="" disabled selected>Pilih Kecamatan</option>`).formSelect();
      W.config.el.desa.empty().append(`<option value="" disabled selected>Pilih Desa</option>`).formSelect();
      W.config.el.kabupaten.empty().append(`<option value="" disabled selected>Pilih Kabupaten/Kota</option>`).formSelect();
      W.config.dataset
        .filter((d) => d.kode.includes($(this).val() + ".") && d.kode.length === 5)
        .forEach((d) => {
          W.config.el.kabupaten.append(`<option value="${d.kode}">${d.nama}</option>`);
        });
      W.config.el.kabupaten.formSelect();
    }
  }
  static #onKabupaten(e) {
    if ($(this).val()) {
      W.config.el.kecamatan.empty().append(`<option value="" disabled selected>Pilih Kecamatan</option>`).formSelect();
      W.config.el.desa.empty().append(`<option value="" disabled selected>Pilih Desa</option>`).formSelect();
      W.config.dataset
        .filter((d) => d.kode.includes($(this).val() + ".") && d.kode.length === 8)
        .forEach((d) => {
          W.config.el.kecamatan.append(`<option value="${d.kode}">${d.nama}</option>`);
        });
      W.config.el.kecamatan.formSelect();
    }
  }
  static #onKecamatan(e) {
    if ($(this).val()) {
      W.config.el.desa.empty().append(`<option value="" disabled selected>Pilih Desa</option>`).formSelect();
      W.config.dataset
        .filter((d) => d.kode.includes($(this).val() + "."))
        .forEach((d) => {
          W.config.el.desa.append(`<option value="${d.kode}">${d.nama}</option>`);
        });
      W.config.el.desa.formSelect();
    }
  }

  static set(params, kabupaten = null, kecamatan = null, desa = null) {
    let provinsi = params;
    if (Array.isArray(provinsi)) {
      provinsi = params[0];
      kabupaten = params[1];
      kecamatan = params[2];
      desa = params[3];
    }
    if (provinsi) {
      this.config.el.provinsi.val(provinsi).trigger("change").formSelect();
    }
    if (kabupaten) {
      this.config.el.kabupaten.val(kabupaten).trigger("change").formSelect();
    }
    if (kecamatan) {
      this.config.el.kecamatan.val(kecamatan).trigger("change").formSelect();
    }
    if (desa) {
      this.config.el.desa.val(desa).formSelect();
    }
  }
}
