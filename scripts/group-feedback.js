// Midterm group feedback form.
// Each submission is saved as one row per rated group member in the Supabase (Postgres)
// table `group_feedback`. BACKEND.key is Supabase's public anon (publishable) key: it is
// meant to be published, and the table's row-level security lets it add rows and nothing
// else, so no one can read submissions through this page.
(() => {
  'use strict';

  const BACKEND = {
    url: '',   // Supabase Project URL, e.g. https://abcdefghijklmnop.supabase.co
    key: '',   // Supabase anon (publishable) key
    table: 'group_feedback'
  };
  const COURSE = 'Demand Planning & Fulfillment';
  const CONTACT = 'chenhao.zhou@rutgers.edu';
  const START_ROWS = 3;
  const MAX_ROWS = 8;

  const form = document.getElementById('feedback-form');
  const rows = document.getElementById('member-rows');
  const addButton = document.getElementById('add-member');
  const submitButton = document.getElementById('submit-button');
  const message = document.getElementById('form-message');
  const thanks = document.getElementById('thanks');
  const evaluatorInput = document.getElementById('evaluator');
  const concernsInput = document.getElementById('concerns');

  const SCORE_OPTIONS = '<option value="">Choose</option>' +
    [0, 1, 2, 3, 4, 5].map(value => `<option value="${value}">${value}</option>`).join('');

  function addRow(focus) {
    if (rows.rows.length >= MAX_ROWS) return;
    const row = document.createElement('tr');
    row.innerHTML =
      '<td data-label="Name"><input type="text" class="member-name" maxlength="100" autocomplete="off" spellcheck="false"></td>' +
      `<td data-label="Score (0 to 5)"><select class="member-score">${SCORE_OPTIONS}</select></td>` +
      '<td data-label="Comments (optional)"><textarea class="member-comment" rows="2" maxlength="1000"></textarea></td>' +
      '<td class="remove-cell"><button type="button" class="remove-row">&times;</button></td>';
    rows.appendChild(row);
    renumber();
    if (focus) row.querySelector('.member-name').focus();
  }

  function renumber() {
    [...rows.rows].forEach((row, index) => {
      const n = index + 1;
      row.querySelector('.member-name').setAttribute('aria-label', `Group member ${n}: name`);
      row.querySelector('.member-score').setAttribute('aria-label', `Group member ${n}: score from 0 to 5`);
      row.querySelector('.member-comment').setAttribute('aria-label', `Group member ${n}: comments (optional)`);
      const remove = row.querySelector('.remove-row');
      remove.setAttribute('aria-label', `Remove group member ${n}`);
      remove.hidden = rows.rows.length === 1;
    });
    addButton.hidden = rows.rows.length >= MAX_ROWS;
  }

  rows.addEventListener('click', event => {
    const button = event.target.closest('.remove-row');
    if (!button) return;
    const row = button.closest('tr');
    const neighbor = row.nextElementSibling || row.previousElementSibling;
    row.remove();
    renumber();
    (neighbor ? neighbor.querySelector('.member-name') : addButton).focus();
  });
  addButton.addEventListener('click', () => addRow(true));

  const squish = text => text.replace(/\s+/g, ' ').trim();

  function read() {
    form.querySelectorAll('[aria-invalid]').forEach(element => element.removeAttribute('aria-invalid'));
    const errors = [];
    const flag = (element, text) => {
      element.setAttribute('aria-invalid', 'true');
      errors.push({ element, text });
    };

    const evaluator = squish(evaluatorInput.value);
    if (!evaluator) flag(evaluatorInput, 'Enter your full name.');

    const members = [];
    const seen = new Set();
    [...rows.rows].forEach((row, index) => {
      const nameInput = row.querySelector('.member-name');
      const scoreInput = row.querySelector('.member-score');
      const name = squish(nameInput.value);
      const comment = row.querySelector('.member-comment').value.trim();
      if (!name && scoreInput.value === '' && !comment) return;
      if (!name) flag(nameInput, `Enter the name of group member ${index + 1}.`);
      if (scoreInput.value === '') flag(scoreInput, `Choose a score for ${name || `group member ${index + 1}`}.`);
      if (name && seen.has(name.toLowerCase())) flag(nameInput, `${name} is listed twice.`);
      seen.add(name.toLowerCase());
      members.push({ name, score: Number(scoreInput.value), comment });
    });
    if (!members.length) flag(rows.querySelector('.member-name'), 'Add at least one group member.');

    return { errors, evaluator, members, concerns: concernsInput.value.trim() };
  }

  function show(text, items) {
    message.replaceChildren(document.createTextNode(text));
    if (items && items.length) {
      const list = document.createElement('ul');
      items.forEach(item => {
        const li = document.createElement('li');
        li.textContent = item;
        list.appendChild(li);
      });
      message.appendChild(list);
    }
    message.hidden = false;
  }

  function uuid() {
    if (window.crypto && crypto.randomUUID) return crypto.randomUUID();
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    const hex = [...bytes].map(byte => byte.toString(16).padStart(2, '0')).join('');
    return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
  }

  async function save(records) {
    const headers = { 'Content-Type': 'application/json', apikey: BACKEND.key, Prefer: 'return=minimal' };
    if (BACKEND.key.startsWith('eyJ')) headers.Authorization = `Bearer ${BACKEND.key}`;
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 20000);
    try {
      const response = await fetch(`${BACKEND.url.replace(/\/+$/, '')}/rest/v1/${BACKEND.table}`, {
        method: 'POST',
        headers,
        body: JSON.stringify(records),
        signal: controller.signal
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}: ${(await response.text()).slice(0, 300)}`);
    } finally {
      clearTimeout(timer);
    }
  }

  form.addEventListener('submit', async event => {
    event.preventDefault();
    const data = read();
    if (data.errors.length) {
      show('Please fix the following:', data.errors.map(error => error.text));
      data.errors[0].element.focus();
      return;
    }
    if (!BACKEND.url || !BACKEND.key) {
      show('This form is not accepting responses yet.');
      return;
    }

    const submissionId = uuid();
    const records = data.members.map(member => ({
      submission_id: submissionId,
      course: COURSE,
      evaluator: data.evaluator,
      teammate: member.name,
      score: member.score,
      comment: member.comment || null,
      concerns: data.concerns || null
    }));

    message.hidden = true;
    submitButton.disabled = true;
    submitButton.textContent = 'Submitting...';
    try {
      await save(records);
      form.hidden = true;
      thanks.hidden = false;
      thanks.focus();
    } catch (error) {
      console.error(error);
      show(`Your feedback was not sent. Check your internet connection and try again. If it still does not go through, email me at ${CONTACT}.`);
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = 'Submit feedback';
    }
  });

  for (let i = 0; i < START_ROWS; i += 1) addRow(false);
})();
