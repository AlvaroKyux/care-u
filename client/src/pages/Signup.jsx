import { useState } from 'react';
import AuthCard from '../components/AuthCard';
import { http } from '../api/http';
import { useAuth } from '../context/AuthContext';
import { Link, useNavigate } from 'react-router-dom';

export default function Signup() {
  const nav = useNavigate();
  const { login } = useAuth();
  const [form, setForm] = useState({ name:'', email:'', password:'', role:'student' });
  const [loading, setLoading] = useState(false);
  const [msg, setMsg] = useState({ type:null, text:'' });

  const onChange = e => setForm(f => ({ ...f, [e.target.name]: e.target.value }));

  async function onSubmit(e){
    e.preventDefault();
    setMsg({ type:null, text:'' });
    setLoading(true);
    try{
      const { data } = await http.post('/auth/register', form);
      login(data.user, data.token);
      nav('/dashboard');
    }catch(err){
      const text = err?.response?.data?.error || 'Error';
      setMsg({ type:'error', text });
    }finally{
      setLoading(false);
    }
  }

  return (
    <AuthCard title="Create your account" subtitle="Sign up to access CARE-U">
      <form onSubmit={onSubmit} className="row" style={{gap:14}}>
        <input className="input" name="name" placeholder="Full name" value={form.name} onChange={onChange} required />
        <input className="input" name="email" type="email" placeholder="Email" value={form.email} onChange={onChange} required />
        <input className="input" name="password" type="password" placeholder="Password (min 6)" value={form.password} onChange={onChange} minLength={6} required />
        <div className="row cols-2">
          <select className="input" name="role" value={form.role} onChange={onChange}>
            <option value="student">Student</option>
            <option value="staff">Staff</option>
            <option value="admin">Admin</option>
          </select>
          <button className="btn" disabled={loading}>{loading ? 'Creating…' : 'Sign up'}</button>
        </div>
        {msg.text && <div className={msg.type === 'error' ? 'error':'success'}>{msg.text}</div>}
      </form>
      <div className="footer-link">
        <span className="helper">Already have an account? </span>
        <Link to="/login">Log in</Link>
      </div>
    </AuthCard>
  );
}
