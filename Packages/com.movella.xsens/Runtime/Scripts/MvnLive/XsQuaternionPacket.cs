///<summary>
/// This class will read the data from the stream and convert it to valid Quaternions.
///</summary>
///<version>
/// 0.1, 2013.03.12, Peter Heinen
/// 1.0, 2013.04.11, Attila Odry
///</version>
///<remarks>
/// Copyright (c) 2013, Xsens Technologies B.V.
/// All rights reserved.
/// 
/// Redistribution and use in source and binary forms, with or without modification,
/// are permitted provided that the following conditions are met:
/// 
/// 	- Redistributions of source code must retain the above copyright notice, 
///		  this list of conditions and the following disclaimer.
/// 	- Redistributions in binary form must reproduce the above copyright notice, 
/// 	  this list of conditions and the following disclaimer in the documentation 
/// 	  and/or other materials provided with the distribution.
/// 
/// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
/// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
/// AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
/// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
/// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, 
/// OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
/// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, 
/// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
///</remarks>
///
using UnityEngine;
using System.IO;
using System;

namespace Movella.Xsens
{
    /// <summary>
    /// Parse the data from the stream as quaternions.
    /// </summary>
    class XsQuaternionPacket : XsDataPacket
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="xsens.XsQuaternionPacket"/> class.
        /// </summary>
        /// <param name='readData'> Create the packet from this data. </param>
        public XsQuaternionPacket(byte[] readData)
            : base(readData)
        {
        }

        // List of vectors with the converted data from the new data packet.
        private Vector3[] convertedData = new Vector3[23];

        /// <summary>
        /// Parses a binary payload to extract segment data, including position and orientation, and handles specific packet types.
        /// </summary>
        /// <param name="br">The BinaryReader used to read the binary stream.</param>
        /// <param name="segmentCount">The number of segments to read from the payload.</param>
        /// <param name="packetType">The type of packet being processed.</param>
        /// <returns>An array of doubles representing the parsed payload data.</returns>
        protected override double[] parsePayload(BinaryReader br, int segmentCount, int packetType)
        {
            // Each segment contains 8 double values (ID, position, quaternion).
            double[] payloadData = new double[segmentCount * 8];

            int startPoint = 0;
            int segmentCounter = 0;

            // Handle data packet type 24: Center of Mass (COM) Data
            if (packetType == 24)
            {
                // Only the three first values of the data packet matter for us (COM position).
                double comX = convert32BitFloat(br.ReadBytes(4));
                double comY = convert32BitFloat(br.ReadBytes(4));
                double comZ = convert32BitFloat(br.ReadBytes(4));

                // Store COM data into the first slot of convertedData, converting the data into the Unity's coordinate system:
                // X in Unity (Position) --> -Y in Xsens (Position)
                // Y in Unity (Position) -->  Z in Xsens (Position)
                // Z in Unity (Position) -->  X in Xsens (Position)
                convertedData[0] = new Vector3(Convert.ToSingle(-comY), Convert.ToSingle(comZ), Convert.ToSingle(comX));
            }

            while (segmentCounter < segmentCount)
            {
                // Read segment data: Body Segment ID (e.g. Right Foot), position (X, Y, Z), quaternion (W, X, Y, Z).
                payloadData[startPoint] = convert32BitInt(br.ReadBytes(4));      // Segment ID
                payloadData[startPoint+1] = convert32BitFloat(br.ReadBytes(4));  // X Position
                payloadData[startPoint+2] = convert32BitFloat(br.ReadBytes(4));  // Y Position
                payloadData[startPoint+3] = convert32BitFloat(br.ReadBytes(4));  // Z Position
                payloadData[startPoint+4] = convert32BitFloat(br.ReadBytes(4));  // Quaternion W
                payloadData[startPoint+5] = convert32BitFloat(br.ReadBytes(4));  // Quaternion X
                payloadData[startPoint+6] = convert32BitFloat(br.ReadBytes(4));  // Quaternion Y
                payloadData[startPoint+7] = convert32BitFloat(br.ReadBytes(4));  // Quaternion Z

                // Handle data packet type 21: Linear Kinematics (we want the linear velocity in the ZXY axes).
                if (packetType == 21)
                {
                    convertedData[segmentCounter] = new Vector3(
                        Convert.ToSingle(-payloadData[startPoint + 5]), // -Linear Velocity X
                        Convert.ToSingle(payloadData[startPoint + 6]),  //  Linear Velocity Y
                        Convert.ToSingle(payloadData[startPoint + 4])   //  Linear Velocity W
                    );

                    br.ReadBytes(8); // Skip 8 bytes (two 4-byte reads)
                }

                // Move to the next segment
                startPoint += 8; // 8 data points per segment
                segmentCounter++;
            }
            return payloadData;
        }

        protected override Vector3[] returnConvertedData() => convertedData;
    } // class XsQuaternionPacket
} // namespace xsens