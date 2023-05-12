# Finite Difference Method（FDM）

> 把变量空间按步长$\Delta x$划分为有限格，从而对微分方程中的微分项进行差分近似，把原方程离散化为代数方程组求解
>
> 1. Euler法：用一阶微分项近似导数项
> 2. Runge-Kutta法：用泰勒展开中的p阶微分项加权近似导数项



## 一阶常微分方程

### Euler法

> 参考：[微分方程数值求解——有限差分法](https://zhuanlan.zhihu.com/p/411798670)

**基本用法**
$$
\left\{
	\begin{aligned}
		u'(x)+c(x)u(x) &= f(x), \quad x \in [a,b] \\
		u(a) &= d
	\end{aligned}
\right.
$$
把$x$所在空间$[a,b]$等分为$N$个子区间，长度为$\Delta x$。当$N$足够大时，微分$u'(x)$的定义可近似为：
$$
\left\{    
	\begin{aligned}        
		u'(x) = \lim\limits_{\Delta x \rightarrow 0}\frac{u(x+\Delta x)-u(x)}{\Delta x} \approx \frac{u(x+\Delta x)-u(x)}{\Delta x}
	\end{aligned}
\right.
$$
$N$越大，精度越高，计算开销越高存在trade-off。

在连续空间$[a,b]$的$u(x)$微分方程转化为离散空间$\{x_i | i=1,2,...,N-1\}$上的未知数$u_i$代数方程组：
$$
\left\{
	\begin{aligned}
    	\frac{1}{\Delta x}(u_{i+1}-u_i)+c(x_i)u_i &= f(x_i), \quad i=0,1,2,...,N-1 \\
        u_0 &= d    
    \end{aligned}
\right.
$$
其中，$x_i = a + i\Delta x$。方程1可以转化为$N \times N$阶矩阵形式：
$$
Au=F: \quad\quad\quad 
	\begin{bmatrix}
		C_0 & 1 &  & \\
		  & C_1 & 1 & \\
		  &  & \ddots & \ddots &  \\
		  &  &  & C_{N-1} & 1 \\
	\end{bmatrix}
	\begin{Bmatrix}
		u_0 \\
		u_1 \\
		\vdots \\
		u_{N-1}
	\end{Bmatrix}
	=
	\begin{Bmatrix}
		F_0 \\
		F_1 \\
		\vdots \\
		F_{N-1}
	\end{Bmatrix}
$$
其中，$C_i=\Delta xc_i-1$，$F_i=\Delta xf_i$。把$u_0=d$的初值条件也引入矩阵，得到$(N+1)\times(N+1)$矩阵：
$$
Au=F: \quad\quad\quad 
	\begin{bmatrix}
		0 & 1 & \cdots & \cdots & 0 \\
		C_0 & 1 &  & & \vdots \\
		\vdots & C_1 & 1 & & \vdots \\
		\vdots &  & \ddots & \ddots & \vdots \\
		0 & \cdots & \cdots & C_{N-1} & 1 \\
	\end{bmatrix}
	\begin{Bmatrix}
		0 \\
		u_0 \\
		u_1 \\
		\vdots \\
		u_{N-1}
	\end{Bmatrix}
	=
	\begin{Bmatrix}
		d \\
		F_0 \\
		F_1 \\
		\vdots \\
		F_{N-1}
	\end{Bmatrix}
$$

该方程组含有$N+1$个方程，$N$个未知数，有唯一解，即为原微分方程在离散格点处的数值解。矩阵$A$满秩，因此可逆，方程解为：$u=A^{-1}F$。

> 验证，取$c(x)=1$，$f(x)=sin(x)+cos(x)$，$x \in [0,2\pi]$，$u(0)=0$：
> $$
> \left \{
> 	\begin{aligned}
> 		u'(x)+u(x) &= sin(x) + cos(x), \quad x \in [0,2\pi] \\
> 		u(0) &= 0
> 	\end{aligned}
> \right.
> $$
> 该方程的解析解为$u(x)=sin(x)$，代码见`Euler1.py`.

**误差分析**

Euler法的误差来自对差分的估计：
$$
\left\{        
	\begin{aligned}
    	u'(x) = \lim\limits_{\Delta x \rightarrow 0}\frac{u(x+\Delta x)-u(x)}{\Delta x} \approx \frac{u(x+\Delta x)-u(x)}{\Delta x}
    \end{aligned}
\right.
$$
实际上$u'(x)$与$\frac{u(x+\Delta x)-u(x)}{\Delta x}$之间差了一个$\Delta x$的二阶小量$\mathcal{O}((\Delta x)^2)$，进而导致积分时每一步都有误差累积，最终数值解与实际解之间的误差为$\Delta x$的一阶小量$\mathcal{O}(\Delta x)$。

> 验证，取$-1$，$f(x)=0$，$x \in [0,10]$，$u(0)=1$：
> $$
> \left \{
> 	\begin{aligned}
>     	u'(x) &= u(x), \quad x \in [0,10] \\
>         u(0) &= 1    
> 	\end{aligned}
> \right.
> $$
> 该方程的解析解为$u(x)=e^x$，代码见`Euler2.py`。会发现随着t增大，数值解与实际值之间的误差越来越大，即误差项累积。

改进的方法是用更高阶的导数项来近似$u'(x)$，从而使误差是$\Delta x$的更高阶小量，Runge-Kutta法就是这样做的。



### Runge-Kutta法

> 参考：[常(偏)微分方程的数值求解（欧拉法、改进欧拉法、龙格-库塔法、亚当姆斯法）](https://zhuanlan.zhihu.com/p/435769998)

**方程形式**
$$
\left\{
	\begin{aligned}
	u'(x) &= f(x, u), \quad x \in [a,b] \\
	u(x_0) &= u_0
	\end{aligned}
\right.
$$
**基本用法**

为了进一步探索降低误差的潜力，即尽可能使误差是$\Delta x$的高阶小量，对求解对象$u(x+\Delta x)$做泰勒展开来观察：
$$
\begin{aligned}
	u(x+\Delta x) &= u(x) + \Delta xu'(x) + \frac{(\Delta x)^2}{2}u''(x) + ... + \frac{(\Delta x)^p}{p!}y^{(p)}(x) + \mathcal{O}((\Delta x)^{p+1}) \\
	&= u(x) + \phi(x) + \mathcal{O}((\Delta x)^{p+1})
\end{aligned}
$$
$\phi(x)$就是满足单步误差为$\Delta x$的$p+1$阶小量的差分近似值（累积误差为$p$阶），对应$p$阶Runge-Kutta法。



#### 一阶Runge-Kutta法（即Euler法）

$$
\begin{aligned}
	u(x+\Delta x) &= u(x) + \Delta xu'(x) + \mathcal{O}((\Delta x)^2) \\
	\phi(x) &= \Delta x u'(x)
\end{aligned}
$$
对一阶微分$u'(x)$取近似，即Euler法中的差分：
$$
\left\{        
	\begin{aligned}
    	u'(x) \approx \phi(x) = \Delta x \frac{u(x+\Delta x)-u(x)}{\Delta x}
    \end{aligned}
\right.
$$
差分的离散形式为：
$$
\begin{aligned}
    u'_i = \frac{u(x_{i+1})-u(x_{i})}{\Delta x}, \quad i=0,1,2,...,N-1
\end{aligned}
$$

从$u(x_0)$到$u(x_{N-1})$进行单步迭代，可解出所有$u(x_{i})$：
$$
\left\{
	\begin{aligned}
	u(x_{i+1}) &= u(x_i)+\Delta x K_1 \\
	K_1 &= f(x_i, u(x_i))
	\end{aligned}
\right.
$$


#### 二阶Runge-Kutta法（即改进的Euler法）

近似到2阶误差：
$$
\begin{aligned}
	u(x+\Delta x) &= u(x) + \Delta xu'(x) + \frac{{\Delta x}^2}{2!}u''(x) + \mathcal{O}((\Delta x)^3) \\
	\phi(x) &= \Delta xu'(x) + \frac{{\Delta x}^2}{2!}u''(x)
\end{aligned}
$$
其中，$u''(x_i)=\frac{u'(x_i+\Delta x) - u'(x_i)}{\Delta x}$，所以：
$$
\phi(x_i) &= \Delta x u'(x_i) + \frac{\Delta x}{2}(u'(x_i+\Delta x) - u'(x_i)) \\
&= \frac{\Delta x}{2}(u'(x_i) + u'(x_i+\Delta x))
$$
先根据公式14中的离散形式差分计算$u'(x_i)$，然后就能代入计算$u'(x_i+\Delta x)$：
$$
\left\{
	\begin{aligned}
	u'(x_i+\Delta x) = f(x_i+\Delta x, u(x_i+\Delta x)) = f(x_i + \Delta x, u(x_i)+\Delta x u'(x_i))
	\end{aligned}
\right.
$$
从$u(x_0)$到$u(x_{N-1})$进行单步迭代，可解出所有$u(x_{i})$：
$$
\left\{
	\begin{aligned}
	u(x_{i+1}) &= u(x_i) + \frac {\Delta x}{2} (K_1 + K_2) \\
	K_1 &= f(x_i, u(x_i)) \\
	K_2 &= f(x_i + \Delta x, u(x_i)+\Delta x K_1)
	\end{aligned}
\right.
$$


#### 四阶Runge-Kutta法（最常用）

$$
\left\{
	\begin{aligned}
	u(x_{i+1}) &= u(x_i) + \frac {\Delta x}{6} (K_1 + 2K_2 + 2K_3 + K_4) \\
	K_1 &= f(x_i, u(x_i)) \\
	K_2 &= f(x_i + \frac{\Delta x}{2}, u(x_i)+\frac{\Delta x}{2} K_1) \\
	K_3 &= f(x_i + \frac{\Delta x}{2}, u(x_i)+\frac{\Delta x}{2} K_2) \\
	K_4 &= f(x_i + \Delta x, u(x_i)+\Delta x K_3)
	\end{aligned}
\right.
$$



#### n阶Runge-Kutta法



<img src="https://my-picture-1311448338.file.myqcloud.com/img/v2-f910858c15413bea8fca740bc906bfa6_r.jpg" alt="img" style="zoom:60%;" />



## 二阶常微分方程



## 一阶偏微分方程



## 一阶随机常微分方程





