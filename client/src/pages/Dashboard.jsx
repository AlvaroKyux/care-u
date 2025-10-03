import { useAuth } from '../context/AuthContext';
import '../styles/theme.css';

export default function Dashboard(){
  const { user, logout } = useAuth();
  if(!user) return <div className="container"><div className="card"><p>Please log in.</p></div></div>;

  return (
    <div className="container">
      <div className="card">
        <h2>Dashboard</h2>
        <p className="helper">Signed in as <b>{user.name}</b> — <i>{user.role}</i></p>

        {user.role === 'admin' && <p>Admin area: manage users, approvals, etc.</p>}
        {user.role === 'staff' && <p>Staff area: tasks, schedules, reports.</p>}
        {user.role === 'student' && <p>Student area: attendance, requests.</p>}

        <div style={{marginTop:16}}>
          <button className="btn" onClick={logout}>Log out</button>
        </div>
      </div>
    </div>
  );
}
