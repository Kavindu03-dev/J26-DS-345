export default function DayTabs({ days, selectedDay, onSelect }) {
  return (
    <nav className="day-tabs" aria-label="Itinerary days">
      {days.map(({ day }) => (
        <button type="button" key={day} className={selectedDay === day ? 'day-tab active' : 'day-tab'}
          aria-pressed={selectedDay === day} onClick={() => onSelect(day)}>
          <span className="day-dot" />Day {day}
        </button>
      ))}
    </nav>
  );
}
