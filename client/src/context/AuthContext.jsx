import { createContext, useContext, useEffect, useState } from 'react';
import { setAuthToken } from '../api/http';

const AuthCtx = createContext(null);

export function AuthProvider({ children }) {
  const [auth, setAuth] = useState(() => {
    const raw = localStorage.getItem('careu_auth');
    return raw ? JSON.parse(raw) : { user: null, token: null };
  });

  useEffect(() => {
    localStorage.setItem('careu_auth', JSON.stringify(auth));
    setAuthToken(auth.token);
  }, [auth]);

  const login = (user, token) => setAuth({ user, token });
  const logout = () => setAuth({ user: null, token: null });

  return (
    <AuthCtx.Provider value={{ ...auth, login, logout }}>
      {children}
    </AuthCtx.Provider>
  );
}

export const useAuth = () => useContext(AuthCtx);
