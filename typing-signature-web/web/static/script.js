(function () {
  const tabPersonal = document.getElementById('tab-personal');
  const tabCorporate = document.getElementById('tab-corporate');
  const panelPersonal = document.getElementById('panel-personal');
  const panelCorporate = document.getElementById('panel-corporate');
  const themeChips = Array.from(document.querySelectorAll('.theme-chip'));
  const btnJoin = document.getElementById('btn-join');

  function activate(tab, panel) {
    for (const el of [tabPersonal, tabCorporate]) {
      el.classList.toggle('is-active', el === tab);
      el.setAttribute('aria-selected', String(el === tab));
    }
    for (const p of [panelPersonal, panelCorporate]) {
      const active = p === panel;
      p.classList.toggle('is-active', active);
      if (active) {
        p.removeAttribute('hidden');
      } else {
        p.setAttribute('hidden', '');
      }
    }
  }

  tabPersonal?.addEventListener('click', () => activate(tabPersonal, panelPersonal));
  tabCorporate?.addEventListener('click', () => activate(tabCorporate, panelCorporate));

  const keystrokeCard = document.getElementById('card-keystroke');
  if (keystrokeCard) {
    keystrokeCard.addEventListener('click', () => {
      window.location.href = 'key_auth.html';
    });
    keystrokeCard.style.cursor = 'pointer';
  }

  if (btnJoin) {
    btnJoin.addEventListener('click', () => {
      window.location.href = 'key_auth.html';
    });
  }

  function setTheme(themeClass) {
    document.body.classList.remove('theme-sunrise', 'theme-honey', 'theme-sunrise-glow', 'theme-sunrise-outline', 'theme-honey-latte');
    document.body.classList.add(themeClass);
    themeChips.forEach((chip) => chip.classList.toggle('is-active', chip.dataset.theme === themeClass));
    try { localStorage.setItem('w24-theme', themeClass); } catch (_) {}
  }
  themeChips.forEach((chip) => {
    chip.addEventListener('click', () => setTheme(chip.dataset.theme));
  });
  try {
    const saved = localStorage.getItem('w24-theme');
    if (saved) setTheme(saved);
  } catch (_) {}
})();


