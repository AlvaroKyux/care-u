import { useAuth } from '../context/AuthContext';
import '../styles/theme.css';
import { Link } from 'react-router-dom';

export default function Dashboard(){
  const { user, logout } = useAuth();
  if(!user) return null;

  return (
    <div className="container">
      <div className="card" style={{minWidth: 420}}>
        <h2>Dashboard</h2>
        <p className="helper">Signed in as <b>{user.name}</b> — <i>{user.role}</i></p>

        {user.role === 'admin' && <p>Admin area: manage users, approvals, etc.</p>}
        {user.role === 'staff' && <p>Staff area: tasks, schedules, reports.</p>}
        {user.role === 'student' && <p>Student area: attendance, requests.</p>}

        {/* Acciones principales */}
        <div style={{display:'grid', gap:12, marginTop:16}}>
          <Link className="btn" to="/new">Report incident</Link>
          <Link className="btn" to="/feed">Open incidents feed</Link>
          <button className="btn" onClick={logout}>Log out</button>
        </div>
      </div>
    </div>
  );
}
