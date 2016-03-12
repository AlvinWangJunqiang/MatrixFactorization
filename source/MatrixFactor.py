# -*- coding=utf-8 -*-
import time
import os
import numpy as np
from scipy import linalg
from numpy import dot

MIN_LIMIT = 1
MAX_LIMIT = 20

class MatrixProcess():
	def __init__(self, file_name, output_file):
		self.speeds = {}
		self.edges = set()
		self.file_name = file_name
		self.output_file = output_file
		#self.file_name = "../data/speed_info/7.csv"
	
	# 非负矩阵分解
	def nmf(self, X, latent_features = 3, max_iter=2000, error_limit=1e-6, fit_error_limit=1e-6):
		"""
		Decompose X to A*Y
		"""
		eps = 1e-5
		print 'Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter)
		#X = X.toarray()  # I am passing in a scipy sparse matrix

		# mask
		mask = np.sign(X)
		rows, columns = X.shape
		A = np.random.rand(rows, latent_features)
		A = np.maximum(A, eps)

		Y = linalg.lstsq(A, X)[0]
		Y = np.maximum(Y, eps)

		masked_X = mask * X
		X_est_prev = dot(A, Y)
		for i in range(1, max_iter + 1):
			top = dot(masked_X, Y.T)
			bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
			A *= top / bottom

			A = np.maximum(A, eps)
			top = dot(A.T, masked_X)
			bottom = dot(A.T, mask * dot(A, Y)) + eps
			Y *= top / bottom
			Y = np.maximum(Y, eps)
			if i % 5 == 0 or i == 1 or i == max_iter:
				print 'Iteration {}:'.format(i),
				X_est = dot(A, Y)
				err = mask * (X_est_prev - X_est)
				fit_residual = np.sqrt(np.sum(err ** 2))
				X_est_prev = X_est
				curRes = linalg.norm(mask * (X - X_est), ord='fro')
				print 'fit residual', np.round(fit_residual, 4),
				print 'total residual', np.round(curRes, 4)
				if curRes < error_limit or fit_residual < fit_error_limit:
					break
		return A, Y

	"""
	@INPUT:
		R     : a matrix to be factorized, dimension N x M
		P     : an initial matrix of dimension N x K
		Q     : an initial matrix of dimension M x K
		K     : the number of latent features
		steps : the maximum number of steps to perform the optimisation
		alpha : the learning rate
		beta  : the regularization parameter
	@OUTPUT:
		the final matrices P and Q
	"""
	def matrix_factorization(self,R, steps=500, alpha=0.0002, beta=0.02):
		N = len(R)
		M = len(R[0])
		K = 2

		P = np.random.rand(N,K)
		Q = np.random.rand(M,K)

		Q = Q.T
		for step in xrange(steps):
			for i in xrange(len(R)):
				for j in xrange(len(R[i])):
					if R[i][j] > 0:
						eij = R[i][j] - np.dot(P[i,:],Q[:,j])
						for k in xrange(K):
							P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
							Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
			eR = np.dot(P,Q)
			e = 0
			for i in xrange(len(R)):
				for j in xrange(len(R[i])):
					if R[i][j] > 0:
						e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
						for k in xrange(K):
							e = e + (beta/2) * ( pow(P[i][k],2) + pow(Q[k][j],2) )
			if e < 0.001:
				break
		return P, Q.T

	# 处理速度文件
	def process(self):
		for index,line in enumerate(open(self.file_name).readlines()):
			if index == 0:
				continue
			records = line.strip().split(',')
			edge_id = records[1]
			dist = float(records[2])
			during = int(records[3])
			courier = records[4]
			timestamp = records[5]
			time_slot = self.getTimeSlot(timestamp)
			if not time_slot:
				continue
			if courier not in self.speeds:
				self.speeds[courier] = {}
			if edge_id not in self.speeds[courier]:
				self.speeds[courier][edge_id] = []
			self.speeds[courier][edge_id].append((dist, during))
			self.edges.add(edge_id)

	# 获取时间戳所在的时间槽
	def getTimeSlot(self, timestamp):
		#return TIME_SLOT_FIRST
		timestamp = int(timestamp[:10])
		ltime = time.localtime(timestamp)
		if ltime.tm_hour >= 9 and ltime.tm_hour <= 12:
			return True
		elif ltime.tm_hour >= 13 and ltime.tm_hour <= 17:
			return True
		else:
			return False
	
	# 生成速度的矩阵
	def generate_matrix(self):
		couriers = self.speeds.keys()
		edges = list(self.edges)
		if len(couriers) == 0:
			return 
		print len(couriers), len(edges)
		speeds_tensor = np.zeros(shape = (len(couriers), len(edges)), dtype = float)
		count = 0
		count_low_speed = 0
		for x in range(len(couriers)):
			courier = couriers[x]
			for y in range(len(edges)):
				edge_id = edges[y]
				if edge_id in self.speeds[courier]: 
					records = self.speeds[courier][edge_id]
					dist = 0
					during = 0
					for record in records:
						if not during > 30:
							dist += record[0]
							during += record[1]
					if (during == 0) :
						pass
					else:
						speed = dist / during
						#if speed > 0.5 and speed < 14:
						if speed > 1:
							speeds_tensor[x][y] = speed
							#print speeds_tensor[x][y]
							count += 1
		print count_low_speed, count, len(couriers) * len(edges), count * 1.0 / (len(couriers) * len(edges))
		print speeds_tensor
		#np.savetxt("../data/output/before.txt", speeds_tensor.ravel())
		nP, nQ = self.matrix_factorization(speeds_tensor)
		#nP, nQ = self.nmf(speeds_tensor)
		#reconstructed = np.dot(nP,nQ)
		reconstructed = np.dot(nP,nQ.T)
		print reconstructed
		#np.savetxt("../data/output/after.txt", reconstructed.ravel())
		writer = open(self.output_file, "a")
		for x in range(len(couriers)):
			courier = couriers[x]
			for y in range(len(edges)):
				edge_id = edges[y]
				speed = reconstructed[x][y]
				if speed > MIN_LIMIT and speed < MAX_LIMIT:
					writer.write("%s,%s,%s\n" % (courier, edge_id, reconstructed[x][y]))
		writer.close()
				

	# 将张量输出
	def output(self, tensor):
		shape = tensor.shape
		writer = open("speed_dense_end.txt", "w")
		writer.write("快递员编号, 道路编号, 时间槽, 速度\n")
		for x in range(shape[0]):
			for y in range(shape[1]):
				courier = self.index_to_courier[y]
				for z in range(shape[2]):
					edge_id = self.index_to_edge[z]
					writer.write("%s,%s,%s,%s\n" % (courier, edge_id, x, tensor[x][y][z]))
		writer.close()

if __name__ == '__main__':
	default_dir = "../data/speed_info_20160124/"
	output_file = "../data/output/dense_speed_20160124_5.csv"
	for file_name in os.listdir(default_dir):
		file_path = default_dir + file_name
		print file_path, output_file
		matrixProcess = MatrixProcess(file_path, output_file)
		matrixProcess.process()
		matrixProcess.generate_matrix()
