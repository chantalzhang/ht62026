const bg = { rock: '#181820', paper: '#2F7D4F', scissors: '#3B1B5E' };
const img = move => move ? `assets/${move}.png` : '';
let lastLockId = 0;
let shownScore = 0;
let compactTimer = null;
let yellTimer = null;
let tauntTimer = null;
let lastCountdownWord = null;
let tauntBag = [];
let playedWinnerTaunt = false;
let lastState = null;

const moveYells = {
  rock: document.getElementById('yellRock'),
  paper: document.getElementById('yellPaper'),
  scissors: document.getElementById('yellScissors'),
};
const firstTaunt = taunt6;
const rockyTaunts = [taunt1, taunt2, taunt3, taunt4, taunt5, taunt7, taunt8];

function playSound(audio) {
  audio.volume = 1;
  audio.currentTime = 0;
  audio.play().catch(() => {});
}

function playMoveYell(move) {
  const audio = moveYells[move];
  if (audio) playSound(audio);
}

function playRockyTaunt() {
  if (!playedWinnerTaunt) {
    playedWinnerTaunt = true;
    playSound(firstTaunt);
    return;
  }
  if (!tauntBag.length) tauntBag = [...rockyTaunts].sort(() => Math.random() - 0.5);
  playSound(tauntBag.pop());
}

async function control(key) {
  await fetch(`/control?key=${encodeURIComponent(key)}`);
}
function resetEndUi() {
  clearTimeout(compactTimer);
  clearTimeout(yellTimer);
  clearTimeout(tauntTimer);
  scoreGroup.classList.remove('compact');
  compactScore.classList.remove('show');
  endActions.classList.remove('show');
  plusOne.classList.remove('show');
}
function animateScore(newScore) {
  score.textContent = String(shownScore).padStart(2, '0');
  setTimeout(() => {
    shownScore = newScore;
    score.textContent = String(shownScore).padStart(2, '0');
    score.classList.remove('bump');
    plusOne.classList.remove('show');
    void score.getBoundingClientRect();
    score.classList.add('bump');
    plusOne.classList.add('show');
  }, 300);
  compactTimer = setTimeout(() => {
    compactScoreText.textContent = `${shownScore} - 0`;
    scoreGroup.classList.add('compact');
    compactScore.classList.add('show');
    endActions.classList.add('show');
  }, 2300);
}

document.addEventListener('keydown', e => {
  if (e.repeat) return;
  if (e.key === ' ') {
    e.preventDefault();
    control('space');
  }
  if (e.key === 'r') control('r');
  if (e.key === 'c') control('c');
  if (e.key === 'q') control('q');
});
giveUpButton.addEventListener('click', () => {
  window.location.href = '/white-flag.html';
});

async function tick() {
  const s = await fetch('/state').then(r => r.json());
  if (!s.locked && s.score !== shownScore) {
    shownScore = s.score;
    score.textContent = String(shownScore).padStart(2, '0');
  }

  const counting = s.state === 'COUNTDOWN';
  scoreGroup.style.display = counting ? 'none' : 'block';
  countdownGroup.style.display = counting ? 'block' : 'none';
  if (s.countdown_word) {
    countdownText.textContent = s.countdown_word;
    if (s.countdown_word !== lastCountdownWord) {
      lastCountdownWord = s.countdown_word;
      playSound(rpsBeep);
    }
  } else {
    if (lastState === 'COUNTDOWN' && s.state !== 'COUNTDOWN') playSound(shootBeep);
    lastCountdownWord = null;
  }
  lastState = s.state;
  rockyFist.classList.toggle('counting', counting);

  if (s.locked && s.lock_id !== lastLockId) {
    lastLockId = s.lock_id;
    resetEndUi();
    animateScore(s.score);
    lockedFrame.setAttribute('href', `/locked_frame?id=${s.lock_id}`);
    shootOverlay.style.display = 'block';
    shootOverlay.classList.remove('play');
    void shootOverlay.getBoundingClientRect();
    shootOverlay.classList.add('play');
    flashOverlay.classList.remove('play');
    void flashOverlay.getBoundingClientRect();
    flashOverlay.classList.add('play');
    setTimeout(() => shootOverlay.style.display = 'none', 1500);
    yellTimer = setTimeout(() => playMoveYell(s.rocky_move), 1550);
    tauntTimer = setTimeout(playRockyTaunt, 2450);
  }

  if (s.locked) {
    rockyFistGroup.style.display = 'none';
    rockyMoveBox.style.display = 'block';
    rockyMove.style.display = 'block';
    rockyMove.setAttribute('href', img(s.rocky_move));
    rockyMoveBox.setAttribute('fill', bg[s.rocky_move]);

    camera.style.display = 'none';
    youMove.style.display = 'block';
    youMove.setAttribute('href', img(s.locked));
    youBox.setAttribute('fill', bg[s.locked]);

    resultPill.style.display = 'block';
    resultText.style.display = 'block';
    resultText.textContent = (s.result || 'Rocky wins').toUpperCase() + '!';
    youText.textContent = `THREW: ${s.locked.toUpperCase()}`;
  } else {
    resetEndUi();
    rockyFistGroup.style.display = 'block';
    rockyMoveBox.style.display = 'none';
    rockyMove.style.display = 'none';

    camera.style.display = 'block';
    youMove.style.display = 'none';
    youBox.setAttribute('fill', '#15151D');

    resultPill.style.display = 'none';
    resultText.style.display = 'none';
    youText.textContent = counting ? s.countdown_word : s.state === 'ARMED' ? `STABLE: ${s.stable}` : 'PRESS SPACE';
  }
}

setInterval(tick, 120);
tick();
