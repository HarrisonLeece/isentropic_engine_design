'''@author Harrison Leece
This program is intended to rapidly prototype liquid bi-propellant engines
using isentropic formulae and some external inputs from NASA CEA.  For our case
a desired Sea Level thrust is required, but optimum is 1/4 MECO to acheive superior delta-V.
This program takes inputs and iterates on a design until Sea Level thrust requirement is met

While the engine is designed in english engineering units for manufacturing purposes,
the thermodynamics are considerably easier when SI units are utilized.  Hence,
pressures and thruts are converted to SI in the __init__

2019/9/8
'''
import numpy as np
#All units should be in english engineering
#ocassionally units will be converted to SI in functions, then back to EE

#Sea Level pressure in pascals
SL_PRES = 101325
#universial gas constant R. 1544ft-lbf/lbm-mol-R or 8314.3 J/kg-mol-K
R_prime_metric = 8314.3

class Engine():
    def __init__(self, target_sl_thrust, optimum_pressure, stag_temp, stag_density, chamber_press, heats_ratio, molar_mass):
        #Input in lbf, converted to Newtons
        self.target_thrust = target_sl_thrust * 4.44822
        #Use CEA to get this!
        self.k = heats_ratio
        #input in psia, converted to pascals
        self.opt_press = optimum_pressure * 6894.76
        #get this from CEA.  Units in Kelvins
        self.stag_temp = stag_temp
        #Design chamber stagnation pressure.  Input in psia, converted to pascals
        self.chm_press = chamber_press * 6894.76
        #Get stagnation_nu from CEA.  Input into Engine. Stores as stagnation specific volume
        self.stag_nu = 1/stag_density
        #molar mass of combustion products.  g/mol 100% need CEA for this one
        self.molar_mass = molar_mass
        #a is the speed of sound at stagnation
        self.a = self.calc_sound_speed_metric(self.k, self.stag_temp, self.molar_mass)

        self.throat_pressure = self.pressure_from_mach(self.k, self.chm_press, 1)
        self.throat_nu = self.nu_from_mach(self.k, self.stag_nu, 1)
        self.throat_temp = self.t_from_mach(self.k, self.stag_temp, 1)
        self.throat_a = self.calc_sound_speed_metric(self.k, self.throat_temp, self.molar_mass)

        self.exit_mach = self.mach_from_pressure(self.k, self.chm_press, self.opt_press)
        self.exit_temp = self.t_from_mach(self.k, self.stag_temp, self.exit_mach)
        self.exit_a = self.calc_sound_speed_metric(self.k, self.exit_temp, self.molar_mass)
        self.exit_velocity = self.calc_exit_velocity(self.k, self.stag_temp, self.molar_mass, self.opt_press, self.chm_press)



        self.solve_engine()

    def pressure_from_mach(self, k, chm_press, mach_number):
        pressure = chm_press / (1+(k-1)/2 * mach_number**2)**(k/(k-1))
        return pressure

    def mach_from_pressure(self, k, chm_press, target_press):
        '''@author Harrison Leece
        @param k The 'gamma' or ratio of specific heats
        @param_chm_press The stagnation pressure of the chamber
        @param target_press The pressure at the location you want the mach number of

        Uses Stagnation to Static pressure ratio for isentropic flow to back out
        mach number.  Intended to circumvent perfect gas laws to identify exit
        plane mach number.  Also allows check-out of CEA results by trying CEA's
        estimation of throat pressure.  At that pressure, M must = 1

        @return mach_at_target_pressure The mach number at a location coresponding to the target pressure
        '''
        first_term = 2/(k-1)
        second_term = ((chm_press/target_press)**((k-1)/k) - 1)
        mach_at_target_pressure = np.sqrt(first_term * second_term)
        return mach_at_target_pressure

    def nu_from_mach(self, k, stagnation_nu, mach_number):
        nu_at_mach = stagnation_nu * (1 + ((k-1)/2)) ** (1/(k-1))
        return nu_at_mach

    def t_from_mach(self, k, stag_t, mach_number):
        temp_at_mach = stag_t/(1 + (k-1)/2 * mach_number**2)
        return temp_at_mach

    def calc_sound_speed(self, k, stag_temp, molar_mass):
        '''@author Harrison Leece
        Converts english units to metric, computes speed of sound at
        certain condtions, converts local speed of sound to feet/s
        Keep this around in case I improve the code to be EE native

        @return local speed of sound in english engineering units
        '''
        #convert Rankine temp to Kelvin
        temp_k = 0.555556*stag_temp
        R = molar_mass * R_prime_metric
        #a_metric is in m/s
        a_metric = np.sqrt(k * R * temp_k)
        #return as feet/s
        return a_metric * 3.28084

    def calc_sound_speed_metric(self, k, temp, molar_mass):
        '''@author Harrison Leece
        Converts english units to metric, computes speed of sound at
        certain condtions, converts local speed of sound to feet/s

        @return local speed of sound in metric units
        '''
        R = R_prime_metric/molar_mass
        #a_metric is in m/s
        a_metric = np.sqrt(k * R * temp)
        #return as m/s
        return a_metric

    def calc_sl_thrust(self, f_opt, exit_area, opt_press):
        '''@author Harrison
        Simple thrust equation.  Check RPE chapter 3, it should be like, the FIRST
        equation in the book
        '''
        return f_opt + exit_area * (opt_press - SL_PRES)

    def calc_throat_area(self, m_dot, throat_nu, throat_speed_of_sound):
        '''@author Harrison Leece
        Density * Area * Velocity = Mass flow rate.  If you know mass flow rate
        you can re-arrange for area and arrive at this equation.  The tricky part
        is calculating speed of sound at the throat conditions.
        '''
        throat_velocity = throat_speed_of_sound
        throat_area = m_dot * throat_nu/throat_velocity
        return throat_area

    def calc_exit_velocity(self, k, stag_temp, molar_mass, opt_press, chm_press):
        '''@author Harrison Leece
        This function calculates the velocity at the exit of the nozzle from
        isentropic thermodynamic relationships.  Based on the mass flow rate
        and exit velocity, optimum expansion ratio thrust can be calculated.
        refer to equation 3-16 in Sutton's RPE, ninth edition

        @return exit_velocity: Nozzle exit velocity
        @todo add entrance velocity if non-negligible
        @todo might need mechanical equivalent of heat factor for English...
              verify with hand calcs
        '''
        first_term = 2*k/(k - 1)
        second_term = R_prime_metric * stag_temp/molar_mass
        third_term = (1 - (opt_press/chm_press)**((self.k - 1)/self.k))
        exit_velocity = np.sqrt(first_term * second_term * third_term)
        return exit_velocity

    def calc_exit_area(self, k, throat_area, exit_mach):
        m2 = exit_mach
        ar_t = throat_area
        exit_area = (ar_t/m2) * np.sqrt(((1+((k-1)/2) * m2**2)/(1+((k-1)/2)))**((k+1)/(k-1)))
        return exit_area


    def solve_engine(self):
        sl_thrust = 0
        m_dot = 0
        while(sl_thrust < self.target_thrust):
            #m_dot is in kg/s
            m_dot += .01
            throat_area = self.calc_throat_area(m_dot, self.throat_nu, self.throat_a)
            exit_area = self.calc_exit_area(self.k, throat_area, self.exit_mach)
            opt_thrust = m_dot * self.exit_velocity
            sl_thrust = self.calc_sl_thrust(opt_thrust, exit_area, self.opt_press)
        print('Sea level thrust (lbf): {}'.format(sl_thrust/4.448))
        print('Optimum thrust (lbf): {}'.format(opt_thrust/4.448))
        print('Mass flow rate required (kg/s): {}'.format(m_dot))
        print('Throat_area (in^2): {}'.format(throat_area * 1550.003))
        print('Exit area (in^2): {}'.format(exit_area * 1550.003))
        print('Specific impulse (s): {}'.format(sl_thrust/(m_dot*9.81)))







if __name__ == '__main__':
    print('Running isentropic engine equations iteratively...')
    #Molar mass from CEA
    molar_mass = 20.4
    engine_design_1 = Engine(3000,8.103,3152,1.7322,325,1.18,molar_mass)
