let edgeThreshold = 50;
const warning = document.getElementById('warning');
const alarmSound = document.getElementById('alarmSound');
const toggleAlarmBtn = document.getElementById('toggleAlarmBtn');
const sensitivityRange = document.getElementById('sensitivityRange');
const sensValueLabel = document.getElementById('sensValue');

let alarmActive = false;
let soundEnabled = false;
let alarmEnabled = true;

document.addEventListener('click', () => {
  if (!soundEnabled) {
    soundEnabled = true;
    console.log("🔊 Sonido habilitado");
  }
});

toggleAlarmBtn.addEventListener('click', () => {
  alarmEnabled = !alarmEnabled;
  if (!alarmEnabled) {
    hideAlarm();
    toggleAlarmBtn.textContent = "Activar Alarma";
  } else {
    toggleAlarmBtn.textContent = "Desactivar Alarma";
  }
});

sensitivityRange.addEventListener('input', () => {
  edgeThreshold = Number(sensitivityRange.value);
  sensValueLabel.textContent = edgeThreshold;
});

document.addEventListener('mousemove', (e) => {
  if (!alarmEnabled) return;
  const x = e.clientX;
  const y = e.clientY;
  const width = window.innerWidth;
  const height = window.innerHeight;
  const isNearEdge =
    x <= edgeThreshold ||
    x >= width - edgeThreshold ||
    y <= edgeThreshold ||
    y >= height - edgeThreshold;
  if (isNearEdge) {
    if (!alarmActive) {
      showAlarm();
    }
  } else {
    hideAlarm();
  }
});

function showAlarm() {
  warning.style.display = 'block';
  if (soundEnabled && alarmEnabled) {
    alarmSound.currentTime = 0;
    alarmSound.play().catch(err => {
      console.warn("⚠️ Error al reproducir sonido:", err);
    });
  }
  alarmActive = true;
}

function hideAlarm() {
  warning.style.display = 'none';
  if (soundEnabled) {
    alarmSound.pause();
    alarmSound.currentTime = 0;
  }
  alarmActive = false;
}