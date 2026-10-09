(() => {
  const nav = document.querySelector('.category-nav');
  const tabs = [...nav.querySelectorAll('a')];
  const panels = [...document.querySelectorAll('.update-panel')];
  const keys = tabs.map(tab => tab.hash.slice(1));

  function select(key, focus = false) {
    const selected = keys.includes(key) ? key : 'extra';
    tabs.forEach(tab => {
      const active = tab.hash === `#${selected}`;
      tab.setAttribute('aria-selected', String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focus) tab.focus();
    });
    panels.forEach(panel => { panel.hidden = panel.id !== selected; });
  }

  nav.setAttribute('role', 'tablist');
  tabs.forEach((tab, index) => {
    tab.setAttribute('role', 'tab');
    tab.setAttribute('aria-controls', keys[index]);
    tab.addEventListener('click', event => {
      event.preventDefault();
      history.pushState(null, '', tab.hash);
      select(keys[index]);
    });
    tab.addEventListener('keydown', event => {
      let next;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      if (event.key === 'ArrowLeft') next = (index - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') next = 0;
      if (event.key === 'End') next = tabs.length - 1;
      if (event.key === ' ') next = index;
      if (next === undefined) return;
      event.preventDefault();
      history.replaceState(null, '', tabs[next].hash);
      select(keys[next], true);
    });
  });
  panels.forEach(panel => {
    panel.setAttribute('role', 'tabpanel');
    panel.setAttribute('aria-labelledby', `tab-${panel.id}`);
    panel.tabIndex = 0;
  });
  window.addEventListener('hashchange', () => select(location.hash.slice(1)));
  window.addEventListener('popstate', () => select(location.hash.slice(1)));
  select(location.hash.slice(1));
})();
