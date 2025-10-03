import jwt from 'jsonwebtoken';

const { JWT_SECRET = 'change_me' } = process.env;

export function signToken(payload, options = {}) {
  return jwt.sign(payload, JWT_SECRET, { expiresIn: '7d', ...options });
}

export function verifyToken(token) {
  return jwt.verify(token, JWT_SECRET);
}
