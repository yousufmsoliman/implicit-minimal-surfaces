#include "singularity_locator.h"

#include "geometrycentral/numerical/linear_algebra_types.h"
#include "geometrycentral/utilities/vector2.h"

namespace minimalmatching
{
    using geometrycentral::DenseMatrix;
    using geometrycentral::SparseMatrix;

    Vector3
    locateSingularity(
        double omega_ij_0,
        double omega_jk_0,
        double omega_ki_0,

        double curvature_ijk_0,

        double z_i, double z_j, double z_k,

        int nSteps
    )
    {
        geometrycentral::Vector<double> x(2);
        x(0) = 1./3.;
        x(1) = 1./3.;

        double dt = 1. / double(nSteps - 1);
        for (int iter = 0; iter < nSteps; ++iter)
        {
            double t = iter * dt;
            double omega_ij = omega_ij_0 + (1.-t) * (curvature_ijk_0 - 2. * omega_ij_0 + omega_jk_0 + omega_ki_0) / 3.;
            double omega_jk = omega_jk_0 + (1.-t) * (curvature_ijk_0 + omega_ij_0 - 2. * omega_jk_0 + omega_ki_0) / 3.;
            double omega_ki = omega_ki_0 + (1.-t) * (curvature_ijk_0 + omega_ij_0 + omega_jk_0 - 2. * omega_ki_0) / 3.;

            double curvature = t * curvature_ijk_0;

            auto F = [z_i, z_j, z_k, omega_ij, omega_jk, omega_ki, curvature](double bj, double bk)
            {
                double q_real = (1. - bj - bk) * z_i;
                q_real += bj * z_j * cos(bk * curvature + omega_ij);
                q_real += bk * z_k * cos(-(bj * curvature + omega_ki));

                double q_imag = 0;
                q_imag += bj * z_j * sin(bj * curvature + omega_ij);
                q_imag += bk * z_k * sin(-(bj * curvature + omega_ki));

                geometrycentral::Vector<double> y(2);
                y(0) = q_real;
                y(1) = q_imag;

                DenseMatrix<double> dF(2,2);

                dF(0,0) = -z_i;
                dF(0,0) += z_j * cos(bk * curvature + omega_ij);
                dF(0,0) += bk * z_k * curvature * sin(-(bj * curvature + omega_ij));

                dF(0,1) = -z_i;
                dF(0,1) -= bj * z_j * curvature * sin(bk * curvature + omega_ki);
                dF(0,1) += z_k * cos(-(bj * curvature + omega_ki));

                dF(1,0) = 0;
                dF(1,0) += z_j * sin(bk * curvature + omega_ij);
                dF(1,0) -= bk * z_k * curvature * cos(-(bj * curvature + omega_ij));

                dF(1,1) = 0;
                dF(1,1) += bj * z_j * curvature * cos(bk * curvature + omega_ki);
                dF(1,1) += z_k * sin(-(bj * curvature + omega_ki));

                return std::make_pair(y, dF);
            };

            auto newton_step = [](geometrycentral::Vector<double> &x, geometrycentral::Vector<double> &Fx, DenseMatrix<double> &dFx)
            {
                DenseMatrix<double> dF_inv(2,2);
                dF_inv(0,0) = dFx(1,1);
                dF_inv(1,1) = dFx(0,0);
                dF_inv(0,1) = -dFx(0,1);
                dF_inv(1,0) = -dFx(1,0);
                dF_inv /= dFx(0,0) * dFx(1,1) - dFx(0,1) * dFx(1,0);

                x -= dF_inv * Fx;
            };

            auto [Fx, dFx] = F(x(0), x(1));
            size_t newton_steps = 0;
            while (Fx.norm() > 1e-4)
            {
                newton_step(x, Fx, dFx);
                std::tie(Fx, dFx) = F(x(0), x(1));

                newton_steps++;
                if (newton_steps > 10)
                {
                    std::cerr << "Failed to find the location of a singularity." << std::endl;
                    break;
                }
            }
        }

        return Vector3{1. - x(0) - x(1), x(0), x(1)};
    }

    Vector2
        locateEdgeEdgeSingularity(
            Vector2 rij,
            Vector2 rxy,

            Vector2 zix, Vector2 zjx,
            Vector2 zjy, Vector2 ziy
        )
    {

        Vector2 b{-1,0};
        b += rij.conj() * (zjx / zix);

        Vector2 c{-1,0};
        c += rxy.conj() * (ziy / zix);

        Vector2 d{1,0};

        Vector2 a{-1,0};
        a += -b - c;
        a += (rij * rxy).conj() * (zjy / zix);

        // Quadratic Equation α t^2 + β t + γ = 0
        double alpha = (a * c.conj()).y;
        double beta = (a * d.conj() - c * b.conj()).y;
        double gamma = (b * d.conj()).y;

        double disc = beta * beta - 4. * alpha * gamma;
        double t0 = (-beta + sqrt(disc)) / (2. * alpha);
        double t1 = (-beta - sqrt(disc)) / (2. * alpha);

        double t = t1;
        if (t0 >= 0 && t0 <= 1)
            t = t0;

        double s = -(c * t + d).x / (a * t + b).x;
        return Vector2{s,t};
    }


}