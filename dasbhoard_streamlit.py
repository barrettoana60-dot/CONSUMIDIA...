:root {
  --bg-blur: 26px;
  --glass-bg: rgba(15, 17, 23, 0.6);
  --glass-border: rgba(255, 255, 255, 0.1);
  --accent: #6cf5ff;
  --accent-soft: rgba(108, 245, 255, 0.18);
  --text-main: #f7f7fb;
  --text-soft: #b4b7c2;
  --danger: #ff6b6b;
  --success: #60e0a4;
  --radius-lg: 18px;
  --radius-md: 12px;
  --radius-sm: 999px;
  --shadow-soft: 0 24px 60px rgba(0, 0, 0, 0.65);
  --font-scale: 1;
}

*,
*::before,
*::after {
  box-sizing: border-box;
}

html,
body {
  margin: 0;
  padding: 0;
  height: 100%;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Text",
    "Segoe UI", sans-serif;
  color: var(--text-main);
  background: #05070a;
}

body {
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  font-size: calc(15px * var(--font-scale));
}

/* BACKGROUND IMAGE / VIDEO STYLE */
.background {
  position: fixed;
  inset: 0;
  background-image: url("https://images.pexels.com/photos/2166553/pexels-photo-2166553.jpeg?auto=compress&cs=tinysrgb&w=1600");
  background-size: cover;
  background-position: center;
  filter: saturate(1.1) contrast(1.05);
  z-index: -2;
}

.background::after {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(
      circle at top left,
      rgba(0, 0, 0, 0.5),
      transparent 50%
    ),
    radial-gradient(circle at bottom right, rgba(0, 0, 0, 0.8), #020308);
  z-index: -1;
}

/* GLASS CARDS */
.glass-card {
  backdrop-filter: blur(var(--bg-blur));
  -webkit-backdrop-filter: blur(var(--bg-blur));
  background: linear-gradient(
      135deg,
      rgba(255, 255, 255, 0.04),
      rgba(0, 0, 0, 0.45)
    );
  border-radius: var(--radius-lg);
  border: 1px solid var(--glass-border);
  box-shadow: var(--shadow-soft);
}

/* AUTH */
.auth-card {
  width: min(420px, 94vw);
  padding: 24px 26px 26px;
  color: var(--text-main);
}

.auth-header h1 {
  margin: 0;
  letter-spacing: 0.14em;
  font-size: 1.7rem;
  text-transform: uppercase;
}

.auth-header p {
  margin: 4px 0 18px;
  color: var(--text-soft);
  font-size: 0.9rem;
}

.auth-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
}

.auth-tab {
  flex: 1;
  padding: 8px 0;
  border-radius: var(--radius-sm);
  border: none;
  background: transparent;
  color: var(--text-soft);
  cursor: pointer;
  font-weight: 500;
}

.auth-tab.active {
  background: var(--accent-soft);
  color: var(--accent);
}

.auth-form {
  display: none;
  display: none;
  flex-direction: column;
  gap: 10px;
}

.auth-form.active {
  display: flex;
}

.auth-form label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.86rem;
  color: var(--text-soft);
}

.auth-form input,
.auth-form select {
  background: rgba(9, 10, 16, 0.7);
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 8px 10px;
  color: var(--text-main);
  outline: none;
}

.auth-form input:focus,
.auth-form select:focus {
  border-color: var(--accent);
}

/* BUTTONS */
.primary-btn,
.ghost-btn {
  border-radius: var(--radius-sm);
  padding: 8px 16px;
  border: none;
  cursor: pointer;
  font-weight: 500;
  font-size: 0.9rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  gap: 6px;
  transition: transform 0.08s ease, background 0.08s ease, box-shadow 0.08s;
}

.primary-btn {
  background: linear-gradient(135deg, #64dcff, #6cf5ff);
  color: #020308;
  box-shadow: 0 8px 30px rgba(75, 228, 255, 0.42);
}

.primary-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 14px 34px rgba(75, 228, 255, 0.5);
}

.ghost-btn {
  background: transparent;
  color: var(--text-soft);
  border: 1px solid rgba(255, 255, 255, 0.14);
}

.ghost-btn:hover {
  background: rgba(255, 255, 255, 0.04);
}

.primary-btn.small,
.ghost-btn.small {
  padding: 6px 12px;
  font-size: 0.78rem;
}

/* APP LAYOUT */
.app {
  width: min(1220px, 95vw);
  height: min(680px, 96vh);
  display: grid;
  grid-template-columns: 230px 1fr;
  gap: 20px;
  color: var(--text-main);
}

.hidden {
  display: none !important;
}

/* SIDEBAR */
.sidebar {
  padding: 16px 14px 14px;
  display: flex;
  flex-direction: column;
  background: linear-gradient(
      160deg,
      rgba(0, 0, 0, 0.8),
      rgba(18, 20, 29, 0.9)
    );
}

.logo {
  display: flex;
  flex-direction: column;
  gap: 2px;
  margin-bottom: 16px;
}

.logo-mark {
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.16em;
  text-transform: uppercase;
}

.logo-text {
  font-size: 0.75rem;
  color: var(--text-soft);
}

.sidebar-nav {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-top: 8px;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  width: 100%;
  padding: 8px 10px;
  border-radius: 12px;
  background: transparent;
  border: none;
  color: var(--text-soft);
  cursor: pointer;
  font-size: 0.84rem;
}

.nav-item .icon {
  width: 18px;
  text-align: center;
}

.nav-item.active {
  background: var(--accent-soft);
  color: var(--accent);
}

.nav-item:hover:not(.active) {
  background: rgba(255, 255, 255, 0.04);
}

.sidebar-footer {
  margin-top: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
  font-size: 0.8rem;
}

.user-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.user-info span {
  font-weight: 500;
}

.user-info small {
  color: var(--text-soft);
}

/* MAIN */
.main {
  padding: 14px 16px;
  display: flex;
  flex-direction: column;
  background: radial-gradient(
      circle at top right,
      rgba(108, 245, 255, 0.08),
      transparent 55%
    ),
    rgba(15, 17, 23, 0.9);
}

/* TOPBAR */
.topbar {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 12px;
}

.top-left h2 {
  margin: 0;
  font-size: 1.1rem;
}

.badge {
  margin-top: 4px;
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.68rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--accent);
  border: 1px solid rgba(108, 245, 255, 0.6);
  background: rgba(108, 245, 255, 0.06);
}

.top-center {
  flex: 1;
}

.search-bar {
  position: relative;
  display: flex;
  align-items: center;
  gap: 6px;
}

.search-bar input {
  flex: 1;
  background: rgba(3, 5, 10, 0.82);
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  padding: 7px 12px;
  color: var(--text-main);
  font-size: 0.84rem;
}

.search-bar input::placeholder {
  color: var(--text-soft);
}

.search-bar button {
  border-radius: 999px;
  border: none;
  padding: 6px 12px;
  font-size: 0.72rem;
  cursor: pointer;
  background: rgba(108, 245, 255, 0.15);
  color: var(--accent);
}

/* VIEWS */
.view {
  flex: 1;
  border-radius: 16px;
  background: linear-gradient(
      145deg,
      rgba(255, 255, 255, 0.02),
      rgba(3, 5, 12, 0.9)
    );
  padding: 14px 12px;
  display: none;
  overflow: hidden;
}

.view.active {
  display: block;
}

.board-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.board-tabs {
  display: flex;
  gap: 6px;
}

.board-tab {
  border-radius: 999px;
  border: none;
  padding: 4px 10px;
  font-size: 0.76rem;
  cursor: pointer;
  background: transparent;
  color: var(--text-soft);
}

.board-tab.active {
  background: rgba(255, 255, 255, 0.08);
  color: var(--text-main);
}

/* KANBAN */
.kanban {
  height: calc(100% - 26px);
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 10px;
}

.kanban-column {
  background: radial-gradient(
      circle at top,
      rgba(255, 255, 255, 0.02),
      transparent 40%
    );
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  padding: 8px;
  display: flex;
  flex-direction: column;
}

.kanban-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 6px;
}

.kanban-header h3 {
  margin: 0;
  font-size: 0.86rem;
}

.pill {
  padding: 1px 8px;
  border-radius: 999px;
  font-size: 0.68rem;
  background: rgba(255, 255, 255, 0.06);
}

.kanban-list {
  flex: 1;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.kanban-card {
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(12, 15, 22, 0.92);
  padding: 6px 7px;
  font-size: 0.76rem;
  cursor: grab;
}

.kanban-card:hover {
  border-color: var(--accent);
}

.kanban-card-title {
  font-weight: 500;
  margin-bottom: 2px;
}

.kanban-card-desc {
  color: var(--text-soft);
  font-size: 0.72rem;
}

.kanban-card-footer {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  font-size: 0.68rem;
  color: var(--text-soft);
}

.add-card-btn {
  margin-top: 6px;
  border-radius: 12px;
  border: 1px dashed rgba(255, 255, 255, 0.16);
  background: rgba(255, 255, 255, 0.02);
  color: var(--text-soft);
  padding: 5px 6px;
  font-size: 0.74rem;
  cursor: pointer;
}

/* SPLIT LAYOUT */
.split {
  display: grid;
  grid-template-columns: 1.2fr 1.1fr;
  gap: 10px;
  height: 100%;
}

.panel {
  background: radial-gradient(
      circle at top,
      rgba(255, 255, 255, 0.02),
      transparent 50%
    );
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.06);
  padding: 10px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.panel.full {
  height: 100%;
}

.panel h3,
.panel h4 {
  margin: 0;
  font-size: 0.96rem;
}

.panel .hint {
  margin: 0;
  font-size: 0.74rem;
  color: var(--text-soft);
}

textarea {
  resize: none;
  min-height: 80px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(3, 5, 10, 0.86);
  padding: 8px;
  color: var(--text-main);
  font-size: 0.84rem;
}

textarea:focus {
  outline: none;
  border-color: var(--accent);
}

input[type="text"],
input[type="date"],
input[type="search"],
select {
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.14);
  background: rgba(3, 5, 10, 0.86);
  padding: 6px 8px;
  color: var(--text-main);
  font-size: 0.84rem;
}

.btn-row {
  display: flex;
  gap: 6px;
}

/* MINDMAP */
.mindmap-container {
  flex: 1;
  overflow: auto;
  padding: 4px;
}

.mindmap {
  list-style: none;
  padding-left: 0;
  margin: 0;
}

.mindmap ul {
  list-style: none;
  padding-left: 18px;
  margin: 2px 0 0;
  border-left: 1px dashed rgba(255, 255, 255, 0.2);
}

.mindmap-node {
  margin: 4px 0;
  position: relative;
}

.mindmap-node .label {
  display: inline-block;
  padding: 3px 8px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.06);
  font-size: 0.78rem;
  cursor: pointer;
}

.mindmap-node.selected .label {
  background: var(--accent-soft);
  color: var(--accent);
}

/* ANALYSIS */
.analysis-output {
  margin-top: 10px;
  padding: 8px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(3, 5, 10, 0.9);
  font-size: 0.84rem;
  max-height: 260px;
  overflow-y: auto;
}

.analysis-output .placeholder {
  margin: 0;
  color: var(--text-soft);
}

.progress-bar {
  margin-top: 8px;
  width: 100%;
  height: 7px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.04);
  overflow: hidden;
}

#progress-fill {
  height: 100%;
  width: 0%;
  border-radius: inherit;
  background: linear-gradient(90deg, #64dcff, #60e0a4);
  transition: width 0.3s ease;
}

#progress-label {
  margin-top: 4px;
  font-size: 0.78rem;
  color: var(--text-soft);
}

/* CHAT */
.chat-messages {
  flex: 1;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(3, 5, 10, 0.9);
  padding: 6px;
  font-size: 0.8rem;
  overflow-y: auto;
}

.chat-message {
  margin-bottom: 6px;
}

.chat-message .meta {
  font-size: 0.7rem;
  color: var(--text-soft);
}

.chat-message .text {
  margin-top: 1px;
}

.chat-form {
  display: flex;
  gap: 6px;
  margin-top: 6px;
}

.chat-form input {
  flex: 1;
}

.thread-list {
  list-style: none;
  padding: 0;
  margin: 0;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(3, 5, 10, 0.9);
  overflow: hidden;
  font-size: 0.8rem;
}

.thread {
  padding: 6px 8px;
  cursor: pointer;
  border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

.thread:last-child {
  border-bottom: none;
}

.thread.active {
  background: var(--accent-soft);
  color: var(--accent);
}

/* NETWORK */
.network-filters {
  display: flex;
  gap: 8px;
  align-items: flex-end;
  margin-bottom: 8px;
}

.network-filters input {
  flex: 1;
}

.network-graph {
  flex: 1;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(3, 5, 10, 0.9);
  padding: 8px;
  font-size: 0.8rem;
  overflow-y: auto;
}

.network-node {
  padding: 5px 6px;
  border-radius: 999px;
  display: inline-block;
  margin: 2px;
  background: rgba(255, 255, 255, 0.06);
}

.network-edge {
  font-size: 0.76rem;
  color: var(--text-soft);
}

/* SETTINGS */
.settings-group {
  margin: 8px 0;
  font-size: 0.86rem;
}

#font-size-range {
  width: 160px;
}

/* MODAL */
.modal {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.6);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 20;
}

.modal-content {
  width: min(360px, 94vw);
  padding: 16px 16px 14px;
}

.modal-content h3 {
  margin-top: 0;
}

.modal-content label {
  display: flex;
  flex-direction: column;
  gap: 4px;
  font-size: 0.8rem;
  margin-top: 8px;
}

.modal-actions {
  margin-top: 12px;
  display: flex;
  justify-content: flex-end;
  gap: 6px;
}

/* HIGH CONTRAST MODE */
body.high-contrast .glass-card,
body.high-contrast .view {
  border-color: rgba(255, 255, 255, 0.6);
}

body.high-contrast .kanban-card {
  border-color: rgba(255, 255, 255, 0.65);
}

/* RESPONSIVE */
@media (max-width: 960px) {
  .app {
    grid-template-columns: 200px 1fr;
  }

  .kanban {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}

@media (max-width: 780px) {
  body {
    align-items: stretch;
  }

  .app {
    grid-template-columns: 1fr;
    grid-template-rows: auto 1fr;
  }

  .sidebar {
    flex-direction: row;
    align-items: center;
    gap: 10px;
    padding: 8px;
  }

  .sidebar-nav {
    flex-direction: row;
    overflow-x: auto;
  }

  .sidebar-footer {
    margin-left: auto;
    align-items: flex-end;
  }

  .kanban {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .split {
    grid-template-columns: 1fr;
  }
}
