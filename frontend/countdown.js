const maxVolume = 1;
crowdCheer.volume = maxVolume;
crowdCheer.play().catch(() => {});

function fadeVolume(from, to, duration) {
  const start = performance.now();
  function step(now) {
    const progress = Math.min((now - start) / duration, 1);
    const eased = progress * progress * progress * (progress * (progress * 6 - 15) + 10);
    crowdCheer.volume = from + (to - from) * eased;
    if (progress < 1) requestAnimationFrame(step);
  }
  requestAnimationFrame(step);
}

setTimeout(() => fadeVolume(maxVolume, 0, 1400), 3400);

setTimeout(() => {
  window.location.href = '/game.html';
}, 5000);
