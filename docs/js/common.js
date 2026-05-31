/* Researcher Population Pyramids — shared layout (header / footer / nav) */

(function () {
  const PAGES = [
    { id: 'home',       href: 'index.html',      label: 'Pyramid' },
    { id: 'trends',     href: 'trends.html',     label: 'Trends' },
    { id: 'comparison', href: 'comparison.html', label: 'Comparison' },
    { id: 'about',      href: 'about.html',      label: 'About' },
  ];

  function injectHeader() {
    const current = document.body.dataset.page || '';
    const headerHTML = `
      <header class="site-header">
        <a class="brand" href="index.html">Researcher Population Pyramids</a>
        <nav>
          ${PAGES.map(p => `<a href="${p.href}"${p.id === current ? ' class="active"' : ''}>${p.label}</a>`).join('')}
        </nav>
      </header>
    `;
    const placeholder = document.getElementById('site-header');
    if (placeholder) {
      placeholder.outerHTML = headerHTML;
    } else {
      document.body.insertAdjacentHTML('afterbegin', headerHTML);
    }
  }

  function injectFooter() {
    const footerHTML = `
      <footer class="site-footer">
        <div>
          Data &amp; method:
          <a href="https://doi.org/10.1093/pnasnexus/pgag059" target="_blank" rel="noopener">Kazuki Nakajima and Takayuki Mizuno (2026), PNAS Nexus</a>.
          Gender inference via
          <a href="https://doi.org/10.1609/icwsm.v17i1.22195" target="_blank" rel="noopener">nomquamgender</a>.
        </div>
        <div class="text-small" style="margin-top: 6px;">
          <a href="about.html">About &amp; Methodology</a> &nbsp;·&nbsp;
          <a href="https://github.com/kazuibasou/researcher_population_pyramids" target="_blank" rel="noopener">Source on GitHub</a>
        </div>
      </footer>
    `;
    const placeholder = document.getElementById('site-footer');
    if (placeholder) {
      placeholder.outerHTML = footerHTML;
    } else {
      document.body.insertAdjacentHTML('beforeend', footerHTML);
    }
  }

  function injectRotateNotice() {
    if (document.querySelector('.rotate-notice')) return;
    const html = `
      <div class="rotate-notice">
        Please rotate your device to landscape for the best viewing experience.
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', html);
  }

  // Public surface
  window.SiteLayout = { injectHeader, injectFooter, injectRotateNotice };

  // Run on DOM ready
  document.addEventListener('DOMContentLoaded', function () {
    injectHeader();
    injectFooter();
    injectRotateNotice();
  });
})();
