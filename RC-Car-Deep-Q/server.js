var fs = require('fs');
var port = 9001;
var app = require('express')();
var https = require('https');
var server = https.createServer({
  key: fs.readFileSync('/etc/'),
  cert: fs.readFileSync('/etc/'),
  ca: fs.readFileSync('/etc/'),
  requestCert: false,
  rejectUnauthorized: false
},app);
server.listen(port, '0.0.0.0');

var io = require('socket.io').listen(server);
var car = io.of('/carsystem');
car.on('connection', function(socket){
  socket.on('carsensor', function(msg){
    socket.broadcast.emit('carsensor', msg);
  });
  socket.on('carupdate', function(msg){
    socket.broadcast.emit('carupdate', msg);
  });
});