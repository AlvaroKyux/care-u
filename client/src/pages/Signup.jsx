import { useState } from 'react';
import AuthCard from '../components/AuthCard';
import { http } from '../api/http';
import { useAuth } from '../context/AuthContext';
import { Link, useNavigate } from 'react-router-dom';

export default function Signup() {
  const nav = useNavigate();
  const { login } = useAuth();
  const [form, setForm] = useState({ name: '', email: '', password: '', role: 'student' }); // Por defecto 'student'
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState({ type: null, text: '' });

  const onChange = (e) => setForm((f) => ({ ...f, [e.target.name]: e.target.value }));

  async function onSubmit(e) {
    e.preventDefault();
    setMsg({ type: null, text: '' });
    setLoading(true);
    try {
      const { data } = await http.post('/auth/register', form);
      login(data.user, data.token);
      nav('/dashboard');
    } catch (err) {
      const api = err?.response?.data?.error;
      const text =
        api === 'Email already registered' ? 'El correo ya está registrado' : api || 'Error del servidor';
      setMsg({ type: 'error', text });
    } finally {
      setLoading(false);
    }
  }

  return (
    <AuthCard title="Crea tu cuenta" subtitle="Regístrate para acceder a CARE-U">
      <form onSubmit={onSubmit} className="row" style={{ gap: 14 }}>
        <input
          className="input"
          name="name"
          placeholder="Nombre completo"
          value={form.name}
          onChange={onChange}
          required
        />
        <input
          className="input"
          name="email"
          type="email"
          placeholder="Correo electrónico"
          value={form.email}
          onChange={onChange}
          required
        />
        <input
          className="input"
          name="password"
          type="password"
          placeholder="Contraseña (mín. 6)"
          value={form.password}
          onChange={onChange}
          minLength={6}
          required
        />
        <div className="row cols-2">
          {/* Campo para seleccionar el rol */}
          <select className="input" name="role" value={form.role} onChange={onChange}>
            <option value="student">Estudiante</option>
            <option value="staff">Personal</option>
            <option value="admin">Administrador</option>
          </select>
          <button className="btn" disabled={loading}>
            {loading ? 'Creando…' : 'Registrarme'}
          </button>
        </div>
        {msg.text && <div className={msg.type === 'error' ? 'error' : 'success'}>{msg.text}</div>}
      </form>
      <div className="footer-link">
        <span className="helper">¿Ya tienes cuenta? </span>
        <Link to="/login">Inicia sesión</Link>
      </div>
    </AuthCard>
  );
}
