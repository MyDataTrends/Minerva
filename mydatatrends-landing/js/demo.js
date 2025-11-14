// Animation logic for the workflow demo

document.addEventListener('DOMContentLoaded', function () {
  const stages = document.querySelectorAll('.stage');
  const startBtn = document.getElementById('start');
  if (!startBtn) return;

  let current = -1;

  function showNext() {
    if (current >= 0) {
      stages[current].classList.remove('active');
    }
    current++;
    if (current < stages.length) {
      stages[current].classList.add('active');
    } else {
      startBtn.disabled = false;
    }
  }

  startBtn.addEventListener('click', () => {
    startBtn.disabled = true;
    current = -1;
    stages.forEach(s => s.classList.remove('active'));
    showNext();
    let idx = 1;
    const interval = setInterval(() => {
      if (idx >= stages.length) {
        clearInterval(interval);
        startBtn.disabled = false;
      } else {
        showNext();
        idx++;
      }
    }, 1500);
  });
});
