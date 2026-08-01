giveUpSound.volume = 0.75;
giveUpSound.play().catch(() => {});
setTimeout(() => {
  window.location.href = '/give-up.html';
}, 3200);
