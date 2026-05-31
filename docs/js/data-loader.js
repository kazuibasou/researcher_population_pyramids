/* Researcher Population Pyramids — JSON loader with memory cache */

(function () {
  const cache = new Map();

  async function fetchJSON(path) {
    if (cache.has(path)) return cache.get(path);
    const url = path.startsWith('http') ? path : path;
    const promise = fetch(url)
      .then(r => {
        if (!r.ok) throw new Error(`HTTP ${r.status} on ${url}`);
        return r.json();
      });
    cache.set(path, promise);
    return promise;
  }

  function dataURL(rel) {
    // Resolve relative to current page so it works at any base path
    // (kazuibasou.github.io/researcher_population_pyramids/...).
    return 'data/' + rel.replace(/^\/+/, '');
  }

  const DataLoader = {
    countries:   () => fetchJSON(dataURL('countries.json')),
    timeline:    () => fetchJSON(dataURL('timeline.json')),
    scatter:     () => fetchJSON(dataURL('scatter.json')),
    pyramid:     (cc) => fetchJSON(dataURL(`pyramids/${cc}.json`)),
    projection:  (cc) => fetchJSON(dataURL(`pyramids_projection/${cc}.json`)),
  };

  window.DataLoader = DataLoader;
})();
