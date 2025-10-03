import { useState } from 'react';
import AuthCard from '../components/AuthCard';
import { http } from '../api/http';
import { useAuth } from '../context/AuthContext';
import { Link, useNavigate } from 'react-router-dom';

export default function Login() {
  const nav = useNavigate();
  const { login } = useAuth();
  const [form, setForm] = useState({ email:'', password:'' });
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState({ type:null, text:'' });

  const onChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  async function onSubmit(e){
    e.preventDefault();
    setMsg({ type:null, text:'' });
    setLoading(true);
    try{
      const { data } = await http.post('/auth/login', form);
      login(data.user, data.token);
      nav('/dashboard');
    }catch(err){
      const api = err?.response?.data?.error;
      const text = api === 'Invalid credentials'
        ? 'Credenciales inválidas'
        : (api || 'Error del servidor');
      setMsg({ type:'error', text });
    }finally{ setLoading(false); }
  }

  return (
    <AuthCard title="Bienvenido" subtitle="Inicia sesión para continuar">
      <form onSubmit={onSubmit} className="row" style={{gap:14}}>
        <input className="input" name="email" type="email" placeholder="Correo electrónico" value={form.email} onChange={onChange} required />
        <input className="input" name="password" type="password" placeholder="Contraseña" value={form.password} onChange={onChange} required />
        <button className="btn" disabled={loading}>{loading ? 'Ingresando…' : 'Iniciar sesión'}</button>
        {msg.text && <div className={msg.type === 'error' ? 'error':'success'}>{msg.text}</div>}
      </form>
      <div className="footer-link">
        <span className="helper">¿No tienes cuenta? </span>
        <Link to="/signup">Regístrate</Link>
      </div>
    </AuthCard>
  );
}
