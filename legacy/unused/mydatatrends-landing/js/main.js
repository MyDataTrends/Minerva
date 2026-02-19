// Simple script to load HTML partials
function loadSection(id, url) {
  fetch(url)
    .then(response => response.text())
    .then(html => {
      document.getElementById(id).innerHTML = html;
    });
}

document.addEventListener('DOMContentLoaded', function () {
  loadSection('hero-section', 'sections/hero.html');
  loadSection('pricing-section', 'sections/pricing.html');
});
