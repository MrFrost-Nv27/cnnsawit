class Puller {
  _datasetprop = function (
    prop = {
      name: "",
      url: "",
      method: "",
      param: "",
      wrap: "",
      data: null,
      callbacks: [],
      done: false,
    }
  ) {
    this.callbacks = prop.callbacks;
    this.name = prop.name;
    this.url = prop.url;
    this.data = prop.data;
    this.method = prop.method;
    this.param = prop.param;
    this.wrap = prop.wrap;
    this.done = prop.done;
  };
  datasets = [];
  constructor() {}

  get(...name) {
    let iterator = [];
    let result = {};
    if (name.length == 0) {
      this.datasets.forEach(ds => {
        result[ds.name] = ds.data;
      });
    }
    if (Array.isArray(name[0])) {
      if (name[0].map((l) => typeof l == "string").includes(false)) return false;
      iterator = name[0];
    } else {
      if (name.map((l) => typeof l == "string").includes(false)) return false;
      iterator = name;
    }

    for (const key of iterator) {
      const r = this.datasets.find((v) => v.name == key) ?? false;
      if (iterator.length == 1) {
        return r ? r.data : r;
      }
      result[key] = r ? r.data : r;
    }

    return result;
  }

  addCallback(name = "", callback = function () {}) {
    if (this.datasets.find((key) => name == key.name)) {
      this.datasets.find((key) => name == key.name).callbacks.push(callback);
    }
    return this;
  }

  async pull(...name) {
    let result = {};
    let iterator = [];
    if (Array.isArray(name[0])) {
      if (name[0].map((l) => typeof l == "string").includes(false)) return false;
      iterator = name[0];
    } else {
      if (name.map((l) => typeof l == "string").includes(false)) return false;
      iterator = name;
    }
    if (undefined == name[0]) iterator = this.datasets.map((d) => d.name);
    for (const key of iterator) {
      if (this.datasets.map((d) => d.name).includes(key)) {
        let item = this.datasets.find((v) => v.name == key);
        item.done = false;
        try {
          item.data = await this.fetch(item.url, {
            method: item.method,
            data: item.param,
            wrap: item.wrap,
          });
          item.callbacks.forEach((cb) => {
            cb(item.data);
          });
        } catch (error) {
          item.data = null;
        }
        item.done = true;
        if (iterator.length == 1) {
          return item.data;
        }
        result[key] = item.data;
      } else {
        result[key] = false;
      }
    }
    return result;
  }

  async add(
    url,
    options = {
      callback: function () {},
      name: null,
      method: "GET",
      data: {},
      wrap: false,
    }
  ) {
    const newDs = new this._datasetprop({
      name: options.name ?? url,
      url: url,
      data: null,
      method: options.method ?? "GET",
      param: options.data ?? {},
      wrap: options.wrap ?? false,
      callbacks: [],
      done: false,
    });
    this.datasets.push(newDs);
    let data = null;
    try {
      data = await this.fetch(url, options);
      newDs.data = data;
      if (options.callback) {
        newDs.callbacks.push(options.callback);
      }
    } catch (error) {
      newDs.data = null;
    }
    newDs.done = true;
    return data;
  }
  async fetch(
    url,
    options = {
      method: "GET",
      data: {},
      wrap: false,
    }
  ) {
    return $.ajax({
      type: options.method ?? "GET",
      url: url,
      data: options.data ?? null,
      success: function (response) {
        return options.wrap ? response[options.wrap] : response;
      },
    });
  }
}
