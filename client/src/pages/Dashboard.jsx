import { useAuth } from '../context/AuthContext';
import '../styles/theme.css';
import { Link } from 'react-router-dom';
import NotificationsBell from '../components/NotificationsBell';

export default function Dashboard(){
  const { user, logout } = useAuth();
  if(!user) return null;

  return (
    <div className="container">
      <div className="card" style={{minWidth: 420}}>
        <h2>Menú</h2>
        <p className="helper">Sesión iniciada como <b>{user.name}</b> — <i>{user.role}</i></p>

        {user.role === 'admin' && <p>Área de administración: usuarios, aprobaciones…</p>}
        {user.role === 'staff' && <p>Área de personal: tareas, horarios, reportes…</p>}
        {user.role === 'student' && <p>Área de estudiante: asistencia, solicitudes…</p>}

        <div style={{display:'grid', gap:12, marginTop:16}}>
          <Link className="btn" to="/new">Reportar incidencia</Link>
          <Link className="btn" to="/feed">Ver feed de incidencias</Link>
          <Link className="btn" to="/profile">Editar perfil</Link>
          <button className="btn" onClick={logout}>Cerrar sesión</button>
        </div>
      </div>

      {/* Campana flotante */}
      <NotificationsBell />
    </div>
  );
}
